# Infrastructure as Code (Terraform)

**이전**: [CLI & SDK](./15_CLI_and_SDK.md) | **다음**: [모니터링, 로깅 & 비용 관리](./17_Monitoring_Logging_Cost.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Infrastructure as Code(코드형 인프라)의 원칙과 수동 프로비저닝 대비 장점을 설명한다
2. 선언적 방식(Terraform, CloudFormation)과 절차적 방식(Ansible)의 IaC 접근법을 비교한다
3. HCL을 사용해 클라우드 리소스를 정의하는 Terraform 구성을 작성한다
4. Terraform 워크플로우(init, plan, apply, destroy)를 실행한다
5. Terraform 상태 파일을 관리하고 팀 협업을 위한 원격 백엔드를 구성한다
6. 변수, 모듈, 출력값을 사용해 재사용 가능하고 유지보수하기 쉬운 인프라 코드를 만든다
7. 적절한 의존성 관리를 갖춘 다중 리소스 배포를 구현한다

---

콘솔에서 리소스를 클릭해서 생성하는 방식은 느리고, 오류가 발생하기 쉬우며, 일관되게 재현하기 어렵습니다. Infrastructure as Code는 클라우드 환경을 소프트웨어처럼 취급합니다 — 버전 관리되고, 코드 리뷰를 거치며, 테스트되고, 자동화된 파이프라인을 통해 배포됩니다. 이는 신뢰할 수 있고, 감사 가능하며, 확장 가능한 클라우드 운영의 토대입니다.

> **비유 — 설계도면, 직접 벽돌 쌓기가 아니라(Blueprints, Not Bricklaying)**: 클라우드 리소스를 수동으로 구성하는 것은 각 작업자에게 구두로 지시하며 집을 짓는 것과 같습니다. Infrastructure as Code는 건축가의 설계도면입니다 — 정확하고 버전 관리된 문서로, 누구나 이를 사용해 동일한 건물을 재현할 수 있습니다. 화재로 집이 소실되더라도 기억에 의존하지 않습니다. 새 팀에게 설계도면을 건네고 동일하게 재건합니다.

## 1. IaC 개요

### 1.1 Infrastructure as Code란?

IaC는 인프라를 코드로 정의하고 관리하는 방식입니다.

**장점:**
- 버전 관리 (Git)
- 재현 가능성
- 자동화
- 문서화
- 협업

### 1.2 IaC 도구 비교

| 도구 | 유형 | 언어 | 멀티 클라우드 |
|------|------|------|-------------|
| **Terraform** | 선언적 | HCL | ✅ |
| CloudFormation | 선언적 | JSON/YAML | AWS만 |
| Deployment Manager | 선언적 | YAML/Jinja | GCP만 |
| Pulumi | 선언적 | Python/TS 등 | ✅ |
| Ansible | 절차적 | YAML | ✅ |

---

## 2. Terraform 기초

### 2.1 설치

```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# 버전 확인
terraform version
```

### 2.2 기본 개념

```
┌─────────────────────────────────────────────────────────────┐
│  Terraform 워크플로우                                        │
│                                                             │
│  1. Write    → .tf 파일 작성                                │
│  2. Init     → terraform init (프로바이더 다운로드)         │
│  3. Plan     → terraform plan (변경 사항 미리보기)          │
│  4. Apply    → terraform apply (인프라 적용)                │
│  5. Destroy  → terraform destroy (인프라 삭제)              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 HCL 문법

```hcl
# 프로바이더 설정
provider "aws" {
  region = "ap-northeast-2"
}

# 리소스 정의
resource "aws_instance" "web" {
  ami           = "ami-12345678"
  instance_type = "t3.micro"

  tags = {
    Name = "WebServer"
  }
}

# 변수
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

# 출력
output "public_ip" {
  value = aws_instance.web.public_ip
}

# 로컬 값
locals {
  environment = "production"
  common_tags = {
    Environment = local.environment
    ManagedBy   = "Terraform"
  }
}

# 데이터 소스 (기존 리소스 참조)
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

## 3. AWS 인프라 구성

### 3.1 VPC + EC2 예제

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

# 퍼블릭 서브넷
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

# 인터넷 게이트웨이
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project}-igw"
  }
}

# 라우팅 테이블
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

# 보안 그룹
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

# EC2 인스턴스
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

# 데이터 소스
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

## 4. GCP 인프라 구성

### 4.1 VPC + Compute Engine 예제

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

# 서브넷
resource "google_compute_subnetwork" "public" {
  name          = "${var.name_prefix}-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.main.id
}

# 방화벽 규칙 - HTTP
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

# 방화벽 규칙 - SSH
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

# Compute Engine 인스턴스
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
      // 외부 IP 할당
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

## 5. 상태 관리

### 5.1 원격 상태 저장소

**AWS S3 백엔드:**
```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "ap-northeast-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"  # 상태 잠금
  }
}
```

**GCP Cloud Storage 백엔드:**
```hcl
terraform {
  backend "gcs" {
    bucket = "my-terraform-state"
    prefix = "prod/terraform.tfstate"
  }
}
```

### 5.2 상태 명령어

```bash
# 상태 목록
terraform state list

# 상태 조회
terraform state show aws_instance.web

# 상태에서 리소스 제거 (실제 리소스는 유지)
terraform state rm aws_instance.web

# 상태 이동 (리팩토링)
terraform state mv aws_instance.old aws_instance.new

# 상태 가져오기 (기존 리소스)
terraform import aws_instance.web i-1234567890abcdef0
```

---

## 6. 모듈

### 6.1 모듈 구조

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

### 6.2 모듈 정의

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

### 6.3 모듈 사용

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

## 7. 워크스페이스

```bash
# 워크스페이스 목록
terraform workspace list

# 새 워크스페이스 생성
terraform workspace new dev
terraform workspace new prod

# 워크스페이스 전환
terraform workspace select prod

# 현재 워크스페이스
terraform workspace show
```

```hcl
# 워크스페이스별 설정
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

## 8. 모범 사례

### 8.1 디렉토리 구조

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

### 8.2 코드 스타일

```hcl
# 리소스 이름 규칙
resource "aws_instance" "web" { }  # 단수
resource "aws_subnet" "public" { } # 복수 사용 시 count/for_each

# 변수 기본값
variable "instance_type" {
  description = "EC2 instance type"  # 항상 설명 포함
  type        = string
  default     = "t3.micro"
}

# 태그 일관성
locals {
  common_tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
```

### 8.3 보안

```hcl
# 민감한 변수
variable "db_password" {
  type      = string
  sensitive = true
}

# 민감한 출력
output "db_password" {
  value     = var.db_password
  sensitive = true
}
```

---

## 9. CI/CD 통합

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

## 10. 다음 단계

- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - 모니터링
- [Docker/](../Docker/) - Kubernetes IaC

---

## 연습 문제

### 연습 문제 1: IaC 도구 선택

팀에서 IaC(Infrastructure as Code) 도구를 선택하고 있습니다. 각 시나리오에 가장 적합한 도구를 평가하세요.

| 시나리오 | 최적 도구 | 이유 |
|---|---|---|
| 하나의 코드베이스로 AWS와 GCP 리소스를 모두 관리 | ? | ? |
| AWS만 사용하며 네이티브 JSON/YAML 지원을 원함 | ? | ? |
| 기존 VM에서 OS 패키지와 소프트웨어를 설정해야 함 | ? | ? |
| HCL 대신 TypeScript로 인프라를 작성하고 싶음 | ? | ? |

<details>
<summary>정답 보기</summary>

| 시나리오 | 최적 도구 | 이유 |
|---|---|---|
| 하나의 코드베이스로 AWS와 GCP 리소스를 모두 관리 | **Terraform** | 멀티 클라우드 지원; 각 클라우드용 프로바이더 플러그인이 있는 단일 HCL 코드베이스 |
| AWS만 사용하며 네이티브 JSON/YAML 지원을 원함 | **CloudFormation** | 네이티브 AWS 서비스; 추가 툴링 불필요; AWS 서비스와 깊은 통합 |
| 기존 VM에서 OS 패키지와 소프트웨어를 설정해야 함 | **Ansible** | 구성 관리를 위해 설계된 절차적 도구 (SSH 기반, 에이전트리스); Terraform은 프로비저닝, Ansible은 구성 담당 |
| HCL 대신 TypeScript로 인프라를 작성하고 싶음 | **Pulumi** | 범용 언어(Python, TypeScript, Go, C#) 지원; 완전한 IDE 지원, 타입 안전성 |

참고: Terraform과 CloudFormation은 모두 선언적(Declarative) 도구입니다 — 원하는 상태를 정의하면 도구가 달성 방법을 결정합니다. Ansible은 절차적(Procedural) 도구입니다 — 순서대로 실행할 단계를 정의합니다. 실제로는 Terraform(프로비저닝) + Ansible(구성)의 조합이 일반적입니다.

</details>

---

### 연습 문제 2: Terraform 워크플로우

S3 버킷과 EC2 인스턴스를 생성하는 Terraform 구성을 작성했습니다. 새 프로젝트를 시작하여 처음으로 구성을 적용하기까지 실행할 정확한 명령어와 각 명령어의 역할을 설명하세요.

<details>
<summary>정답 보기</summary>

```bash
# 1단계: 작업 디렉토리 초기화
# 프로바이더 플러그인(hashicorp/aws)을 다운로드하고 백엔드를 설정합니다
terraform init

# 2단계: 구성 구문 유효성 검사
# 실제 인프라에 접촉하기 전에 HCL 구문 오류와 타입 불일치를 감지합니다
terraform validate

# 3단계: 변경사항 미리보기 (드라이 런)
# 생성/수정/삭제될 리소스를 표시합니다; 아무것도 변경하지 않습니다
terraform plan

# 4단계: 구성 적용
# 실제 리소스를 생성합니다; -auto-approve를 사용하지 않으면 확인을 요청합니다
terraform apply

# 선택사항: 이 구성으로 관리되는 모든 리소스 삭제
terraform destroy
```

**`terraform apply` 중에 내부적으로 일어나는 일:**
1. 현재 상태(`.tfstate` 파일 또는 원격 백엔드)를 읽음
2. AWS API를 호출하여 현재 실제 상태 확인
3. 원하는 상태(`.tf` 파일)와 실제 상태의 차이를 계산
4. 올바른 의존성 순서로 리소스를 생성/업데이트/삭제하는 API 호출 실행
5. 새 상태를 백엔드에 기록

중요: `.tfstate` 파일을 수동으로 편집하지 마세요 — 대신 `terraform state` 명령어를 사용하세요. 상태 파일 손상은 가장 흔한 Terraform 재앙입니다.

</details>

---

### 연습 문제 3: 원격 상태(Remote State)와 잠금(Locking)

3명의 엔지니어로 구성된 팀이 동일한 프로덕션 인프라에서 모두 `terraform apply`를 실행합니다. 어떤 문제가 발생할 수 있으며, 이를 방지하기 위한 상태 잠금이 포함된 원격 백엔드를 어떻게 구성하나요?

<details>
<summary>정답 보기</summary>

**팀 환경에서 로컬 상태의 문제점:**
- **상태 충돌**: 두 엔지니어가 동시에 적용 → 둘 다 같은 이전 상태를 읽음 → 둘 다 같은 변경을 한다고 생각 → 한 명이 다른 사람의 상태를 덮어씀, 드리프트 발생
- **상태 손실**: 상태 파일이 한 노트북에만 존재 → 엔지니어가 떠나거나 노트북이 고장 → 상태 손실 → Terraform이 더 이상 관리 대상을 인식하지 못함
- **이력 없음**: 누가 언제 무엇을 변경했는지 감사 추적 없음

**해결책: 상태 잠금이 있는 원격 백엔드**

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "mycompany-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "ap-northeast-2"
    encrypt        = true                        # 저장 시 상태 암호화
    dynamodb_table = "terraform-state-locks"     # 잠금 테이블
  }
}
```

**DynamoDB 잠금 테이블 생성 (일회성 설정):**
```bash
aws dynamodb create-table \
    --table-name terraform-state-locks \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST
```

**잠금 작동 방식:**
1. 엔지니어 A가 `terraform apply` 실행 → DynamoDB에서 잠금 획득 (LockID 항목 쓰기)
2. 엔지니어 B가 동시에 `terraform apply` 실행 → 잠금 획득 시도 → 잠금 보유 중 → Terraform이 오류로 종료: "Error acquiring the state lock"
3. 엔지니어 A가 완료 → 잠금 해제
4. 엔지니어 B가 이제 안전하게 진행 가능

**GCP의 경우, Cloud Storage 백엔드 사용 (잠금은 객체 버전 관리를 통해 내장됨):**
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

### 연습 문제 4: 변수(Variable)와 출력(Output)

RDS 인스턴스에 대한 다음 Terraform 리소스가 있습니다. 설명이 포함된 변수를 사용하도록 리팩토링하고, 데이터베이스 엔드포인트와 포트를 위한 출력을 추가하세요.

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
<summary>정답 보기</summary>

```hcl
# variables.tf
variable "db_identifier" {
  description = "RDS 인스턴스의 이름"
  type        = string
  default     = "myapp-db"
}

variable "db_instance_class" {
  description = "RDS 인스턴스 유형"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "할당된 스토리지 크기 (GB)"
  type        = number
  default     = 20
}

variable "db_username" {
  description = "데이터베이스 마스터 사용자 이름"
  type        = string
  default     = "admin"
}

variable "db_password" {
  description = "데이터베이스 마스터 비밀번호"
  type        = string
  sensitive   = true  # plan/apply 출력이나 state list에 절대 표시되지 않음
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
  description = "RDS 인스턴스의 연결 엔드포인트"
  value       = aws_db_instance.main.endpoint
}

output "db_port" {
  description = "데이터베이스가 수신 대기 중인 포트"
  value       = aws_db_instance.main.port
}
```

```bash
# 환경 변수를 통해 민감한 비밀번호 설정 (terraform.tfvars에는 절대 넣지 말 것)
export TF_VAR_db_password="SuperSecret123!"
terraform apply
```

핵심 포인트:
- 로그와 CLI 출력에서 비밀번호가 삭제되도록 `sensitive = true`로 표시하세요
- 버전 관리에 커밋되는 `.tf` 파일에는 자격 증명을 하드코딩하지 마세요
- 민감한 값은 `TF_VAR_` 환경 변수 또는 시크릿 매니저 통합(예: Vault 프로바이더)을 통해 전달하세요

</details>

---

### 연습 문제 5: Terraform 모듈(Module) 설계

동일한 3-티어 아키텍처(VPC + 웹 티어 + 데이터베이스 티어)를 `dev`와 `prod` 환경 모두에 배포해야 합니다. `dev` 환경은 더 작고 저렴한 인스턴스를 사용합니다. 모듈과 디렉토리 구조를 설명하고, `prod/main.tf`가 `dev/main.tf`와 다르게 모듈을 호출하는 방법을 보여주세요.

<details>
<summary>정답 보기</summary>

**디렉토리 구조:**
```
terraform/
├── modules/
│   ├── vpc/
│   │   ├── main.tf       # VPC, 서브넷, IGW, 라우팅 테이블
│   │   ├── variables.tf  # cidr_block, name 등
│   │   └── outputs.tf    # vpc_id, public_subnet_ids, private_subnet_ids
│   ├── web/
│   │   ├── main.tf       # EC2 + Auto Scaling Group + ALB
│   │   ├── variables.tf  # instance_type, subnet_ids 등
│   │   └── outputs.tf    # alb_dns_name
│   └── database/
│       ├── main.tf       # RDS 인스턴스 + 서브넷 그룹 + 보안 그룹
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
  instance_type = "t3.micro"        # 저렴한 dev 인스턴스
  subnet_ids    = module.vpc.public_subnet_ids
}

module "database" {
  source            = "../../modules/database"
  db_instance_class = "db.t3.micro" # 가장 작은 RDS
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
  instance_type = "t3.large"        # 프로덕션급 인스턴스
  subnet_ids    = module.vpc.public_subnet_ids
}

module "database" {
  source            = "../../modules/database"
  db_instance_class = "db.r6g.large"  # 프로덕션 RDS
  multi_az          = true             # 고가용성(High Availability)
  subnet_ids        = module.vpc.private_subnet_ids
}
```

이 구조의 장점:
- 모듈 코드는 한 번 작성하고 환경 간에 재사용 — DRY 원칙
- 환경별 파라미터는 각 환경의 `main.tf`와 `terraform.tfvars`에 격리
- 환경별 다른 Terraform 상태 파일로 dev의 `destroy`가 prod에 영향을 주지 않음
- 새 환경(예: `staging`) 추가는 새 디렉토리를 만들고 기존 모듈을 호출하기만 하면 됨

</details>

---

## 참고 자료

- [Terraform Documentation](https://www.terraform.io/docs)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
