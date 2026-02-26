# VPC (Virtual Private Cloud)

**이전**: [블록 및 파일 스토리지](./08_Block_and_File_Storage.md) | **다음**: [로드밸런싱 & CDN](./10_Load_Balancing_CDN.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 가상 사설 클라우드(Virtual Private Cloud)의 목적과 아키텍처를 설명할 수 있습니다
2. AWS VPC(리전 범위)와 GCP VPC(글로벌 범위) 및 각각의 서브넷 모델을 비교할 수 있습니다
3. 적절한 CIDR 블록을 사용하여 퍼블릭 및 프라이빗 서브넷이 포함된 VPC를 설계할 수 있습니다
4. 트래픽 라우팅을 위한 라우팅 테이블(Route Table), 인터넷 게이트웨이(Internet Gateway), NAT 게이트웨이(NAT Gateway)를 구성할 수 있습니다
5. 보안 그룹(Security Group)과 네트워크 ACL(Network ACL)을 적용하여 인바운드 및 아웃바운드 트래픽을 제어할 수 있습니다
6. 네트워크 간 통신을 위한 VPC 피어링(VPC Peering)과 VPN 연결을 구현할 수 있습니다
7. 퍼블릭, 프라이빗, 격리된 서브넷 패턴을 구분할 수 있습니다

---

네트워킹은 모든 클라우드 배포의 기반입니다. VPC는 IP 주소 지정, 서브넷, 라우팅, 접근 규칙을 직접 제어할 수 있는 격리된 소프트웨어 정의 네트워크를 제공합니다. 잘못 구성된 네트워킹은 클라우드에서 발생하는 보안 사고와 장애의 주요 원인 중 하나이므로, 프로덕션 워크로드를 배포하기 전에 VPC 설계를 이해하는 것이 필수적입니다.

> **비유 — 공유 호텔의 전용 층**: 퍼블릭 클라우드는 대형 호텔과 같습니다. VPC 없이는 모든 투숙객이 같은 복도를 공유하고 어떤 방이든 두드릴 수 있습니다. VPC는 전용 엘리베이터 키가 있는 프라이빗 층 전체를 예약하는 것과 같습니다. 해당 층의 투숙객은 방(서브넷) 사이를 자유롭게 이동할 수 있지만, 외부인은 보안(보안 그룹과 NACL)을 통과한 후 프런트 데스크(인터넷 게이트웨이)를 통해서만 입장할 수 있습니다.

## 1. VPC 개요

### 1.1 VPC란?

VPC는 클라우드 내에서 논리적으로 격리된 가상 네트워크입니다.

**핵심 개념:**
- 자체 IP 주소 범위 정의
- 서브넷으로 분할
- 라우팅 테이블로 트래픽 제어
- 보안 그룹/방화벽으로 접근 통제

### 1.2 AWS vs GCP VPC 차이

| 항목 | AWS VPC | GCP VPC |
|------|---------|---------|
| **범위** | 리전 단위 | **글로벌** |
| **서브넷 범위** | 가용 영역 (AZ) | 리전 |
| **기본 VPC** | 리전당 1개 | 프로젝트당 1개 (default) |
| **피어링** | 리전 간 가능 | 글로벌 자동 |
| **IP 범위** | 생성 시 고정 | 서브넷 추가 가능 |

```
AWS VPC 구조:
┌──────────────────────────────────────────────────────────────┐
│  VPC (리전: ap-northeast-2)                                  │
│  CIDR: 10.0.0.0/16                                           │
│  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │ Subnet-a (AZ-a)     │  │ Subnet-b (AZ-b)     │            │
│  │ 10.0.1.0/24         │  │ 10.0.2.0/24         │            │
│  └─────────────────────┘  └─────────────────────┘            │
└──────────────────────────────────────────────────────────────┘

GCP VPC 구조:
┌──────────────────────────────────────────────────────────────┐
│  VPC (글로벌)                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │ Subnet-asia         │  │ Subnet-us           │            │
│  │ (asia-northeast3)   │  │ (us-central1)       │            │
│  │ 10.0.1.0/24         │  │ 10.0.2.0/24         │            │
│  └─────────────────────┘  └─────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 서브넷

### 2.1 퍼블릭 vs 프라이빗 서브넷

| 유형 | 인터넷 접근 | 용도 |
|------|-----------|------|
| **퍼블릭** | 직접 가능 | 웹 서버, Bastion |
| **프라이빗** | NAT 통해서만 | 애플리케이션, DB |

### 2.2 AWS 서브넷 생성

```bash
# 1. VPC 생성
aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=MyVPC}]'

# 2. 퍼블릭 서브넷 생성
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.1.0/24 \
    --availability-zone ap-northeast-2a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Public-Subnet-1}]'

# 3. 프라이빗 서브넷 생성
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.10.0/24 \
    --availability-zone ap-northeast-2a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Private-Subnet-1}]'

# 4. 퍼블릭 IP 자동 할당 (퍼블릭 서브넷)
aws ec2 modify-subnet-attribute \
    --subnet-id subnet-public \
    --map-public-ip-on-launch
```

### 2.3 GCP 서브넷 생성

```bash
# 1. 커스텀 모드 VPC 생성
gcloud compute networks create my-vpc \
    --subnet-mode=custom

# 2. 서브넷 생성 (서울)
gcloud compute networks subnets create subnet-asia \
    --network=my-vpc \
    --region=asia-northeast3 \
    --range=10.0.1.0/24

# 3. 서브넷 생성 (미국)
gcloud compute networks subnets create subnet-us \
    --network=my-vpc \
    --region=us-central1 \
    --range=10.0.2.0/24

# 4. 프라이빗 Google 액세스 활성화
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-private-ip-google-access
```

---

## 3. 인터넷 게이트웨이

### 3.1 AWS Internet Gateway

```bash
# 1. IGW 생성
aws ec2 create-internet-gateway \
    --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=MyIGW}]'

# 2. VPC에 연결
aws ec2 attach-internet-gateway \
    --internet-gateway-id igw-12345678 \
    --vpc-id vpc-12345678

# 3. 라우팅 테이블에 경로 추가
aws ec2 create-route \
    --route-table-id rtb-12345678 \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id igw-12345678

# 4. 퍼블릭 서브넷에 라우팅 테이블 연결
aws ec2 associate-route-table \
    --route-table-id rtb-12345678 \
    --subnet-id subnet-public
```

### 3.2 GCP 인터넷 접근

GCP는 별도의 인터넷 게이트웨이 없이 외부 IP가 있으면 인터넷 접근이 가능합니다.

```bash
# 외부 IP 할당 (인스턴스 생성 시)
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --network=my-vpc \
    --subnet=subnet-asia \
    --address=EXTERNAL_IP  # 또는 생략하면 임시 IP 할당

# 정적 IP 예약
gcloud compute addresses create my-static-ip \
    --region=asia-northeast3
```

---

## 4. NAT Gateway

프라이빗 서브넷의 인스턴스가 인터넷에 접근할 수 있도록 합니다.

### 4.1 AWS NAT Gateway

```bash
# 1. Elastic IP 할당
aws ec2 allocate-address --domain vpc

# 2. NAT Gateway 생성 (퍼블릭 서브넷에)
aws ec2 create-nat-gateway \
    --subnet-id subnet-public \
    --allocation-id eipalloc-12345678 \
    --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=MyNAT}]'

# 3. 프라이빗 라우팅 테이블에 경로 추가
aws ec2 create-route \
    --route-table-id rtb-private \
    --destination-cidr-block 0.0.0.0/0 \
    --nat-gateway-id nat-12345678

# 4. 프라이빗 서브넷에 라우팅 테이블 연결
aws ec2 associate-route-table \
    --route-table-id rtb-private \
    --subnet-id subnet-private
```

### 4.2 GCP Cloud NAT

```bash
# 1. Cloud Router 생성
gcloud compute routers create my-router \
    --network=my-vpc \
    --region=asia-northeast3

# 2. Cloud NAT 생성
gcloud compute routers nats create my-nat \
    --router=my-router \
    --region=asia-northeast3 \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges
```

---

## 5. 보안 그룹 / 방화벽

### 5.1 AWS Security Groups

보안 그룹은 인스턴스 레벨의 **상태 저장(stateful)** 방화벽입니다.

```bash
# 보안 그룹 생성
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server security group" \
    --vpc-id vpc-12345678

# 인바운드 규칙 추가
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 22 \
    --cidr 203.0.113.0/24  # 특정 IP만 허용

# 다른 보안 그룹에서 오는 트래픽 허용
aws ec2 authorize-security-group-ingress \
    --group-id sg-db \
    --protocol tcp \
    --port 3306 \
    --source-group sg-web

# 규칙 조회
aws ec2 describe-security-groups --group-ids sg-12345678
```

### 5.2 GCP Firewall Rules

GCP 방화벽 규칙은 VPC 레벨에서 작동하며 **태그** 또는 서비스 계정으로 대상을 지정합니다.

```bash
# HTTP 트래픽 허용 (태그 기반)
gcloud compute firewall-rules create allow-http \
    --network=my-vpc \
    --allow=tcp:80,tcp:443 \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0

# SSH 허용 (특정 IP)
gcloud compute firewall-rules create allow-ssh \
    --network=my-vpc \
    --allow=tcp:22 \
    --target-tags=ssh-server \
    --source-ranges=203.0.113.0/24

# 내부 통신 허용
gcloud compute firewall-rules create allow-internal \
    --network=my-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/8

# 규칙 목록 조회
gcloud compute firewall-rules list --filter="network:my-vpc"

# 규칙 삭제
gcloud compute firewall-rules delete allow-http
```

### 5.3 AWS NACL (Network ACL)

NACL은 서브넷 레벨의 **상태 비저장(stateless)** 방화벽입니다.

```bash
# NACL 생성
aws ec2 create-network-acl \
    --vpc-id vpc-12345678 \
    --tag-specifications 'ResourceType=network-acl,Tags=[{Key=Name,Value=MyNACL}]'

# 인바운드 규칙 추가 (규칙 번호로 우선순위)
aws ec2 create-network-acl-entry \
    --network-acl-id acl-12345678 \
    --ingress \
    --rule-number 100 \
    --protocol tcp \
    --port-range From=80,To=80 \
    --cidr-block 0.0.0.0/0 \
    --rule-action allow

# 아웃바운드 규칙도 필요 (stateless)
aws ec2 create-network-acl-entry \
    --network-acl-id acl-12345678 \
    --egress \
    --rule-number 100 \
    --protocol tcp \
    --port-range From=1024,To=65535 \
    --cidr-block 0.0.0.0/0 \
    --rule-action allow
```

---

## 6. VPC 피어링

### 6.1 AWS VPC Peering

```bash
# 1. 피어링 연결 요청
aws ec2 create-vpc-peering-connection \
    --vpc-id vpc-requester \
    --peer-vpc-id vpc-accepter \
    --peer-region ap-northeast-1  # 다른 리전인 경우

# 2. 피어링 연결 수락
aws ec2 accept-vpc-peering-connection \
    --vpc-peering-connection-id pcx-12345678

# 3. 양쪽 VPC의 라우팅 테이블에 경로 추가
# Requester VPC
aws ec2 create-route \
    --route-table-id rtb-requester \
    --destination-cidr-block 10.1.0.0/16 \
    --vpc-peering-connection-id pcx-12345678

# Accepter VPC
aws ec2 create-route \
    --route-table-id rtb-accepter \
    --destination-cidr-block 10.0.0.0/16 \
    --vpc-peering-connection-id pcx-12345678
```

### 6.2 GCP VPC Peering

```bash
# 1. 첫 번째 VPC에서 피어링 생성
gcloud compute networks peerings create peer-vpc1-to-vpc2 \
    --network=vpc1 \
    --peer-network=vpc2

# 2. 두 번째 VPC에서 피어링 생성 (양쪽 필요)
gcloud compute networks peerings create peer-vpc2-to-vpc1 \
    --network=vpc2 \
    --peer-network=vpc1

# 라우팅은 자동으로 추가됨
```

---

## 7. 프라이빗 엔드포인트

인터넷을 거치지 않고 AWS/GCP 서비스에 접근합니다.

### 7.1 AWS VPC Endpoints

**Gateway Endpoint (S3, DynamoDB):**
```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-12345678 \
    --service-name com.amazonaws.ap-northeast-2.s3 \
    --route-table-ids rtb-12345678
```

**Interface Endpoint (다른 서비스):**
```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-12345678 \
    --service-name com.amazonaws.ap-northeast-2.secretsmanager \
    --vpc-endpoint-type Interface \
    --subnet-ids subnet-12345678 \
    --security-group-ids sg-12345678
```

### 7.2 GCP Private Service Connect

```bash
# Private Google Access 활성화
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-private-ip-google-access

# Private Service Connect 엔드포인트
gcloud compute addresses create psc-endpoint \
    --region=asia-northeast3 \
    --subnet=subnet-asia \
    --purpose=PRIVATE_SERVICE_CONNECT
```

---

## 8. 일반적인 VPC 아키텍처

### 8.1 3티어 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  VPC (10.0.0.0/16)                                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Public Subnets (10.0.1.0/24, 10.0.2.0/24)               ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │    ALB      │  │   Bastion   │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Private Subnets - App (10.0.10.0/24, 10.0.11.0/24)      ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │   App-1     │  │   App-2     │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Private Subnets - DB (10.0.20.0/24, 10.0.21.0/24)       ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │  DB Primary │  │  DB Standby │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │     IGW      │  │   NAT GW     │                         │
│  └──────────────┘  └──────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 AWS VPC 예시 (Terraform)

```hcl
# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "main-vpc" }
}

# Public Subnets
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "public-${count.index + 1}" }
}

# Private Subnets
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "private-${count.index + 1}" }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

# NAT Gateway
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
}
```

---

## 9. 문제 해결

### 9.1 연결 문제 체크리스트

```
□ 보안 그룹 인바운드 규칙 확인
□ NACL 규칙 확인 (AWS)
□ 방화벽 규칙 확인 (GCP)
□ 라우팅 테이블 확인
□ 인터넷 게이트웨이 연결 확인
□ NAT 게이트웨이 상태 확인
□ 인스턴스에 퍼블릭 IP 있는지 확인
□ VPC 피어링 라우팅 확인
```

### 9.2 디버깅 명령어

**AWS:**
```bash
# VPC Flow Logs 활성화
aws ec2 create-flow-logs \
    --resource-type VPC \
    --resource-ids vpc-12345678 \
    --traffic-type ALL \
    --log-destination-type cloud-watch-logs \
    --log-group-name vpc-flow-logs

# Reachability Analyzer
aws ec2 create-network-insights-path \
    --source i-source \
    --destination i-destination \
    --destination-port 80 \
    --protocol tcp
```

**GCP:**
```bash
# VPC Flow Logs 활성화
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-flow-logs

# Connectivity Tests
gcloud network-management connectivity-tests create my-test \
    --source-instance=projects/PROJECT/zones/ZONE/instances/source \
    --destination-instance=projects/PROJECT/zones/ZONE/instances/dest \
    --destination-port=80 \
    --protocol=TCP
```

---

## 10. 다음 단계

- [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) - 로드밸런싱
- [14_Security_Services.md](./14_Security_Services.md) - 보안 상세

---

## 연습 문제

### 연습 문제 1: CIDR 블록(CIDR Block) 계획

회사가 3-계층(tier) 웹 애플리케이션(웹, 애플리케이션, 데이터베이스)을 위한 AWS VPC를 설정하고 있습니다. `ap-northeast-2` 리전(3개 AZ: a, b, c)에서 VPC CIDR 블록 `10.100.0.0/16`을 사용할 예정입니다.

서브넷 레이아웃을 설계하세요: 3개의 퍼블릭 서브넷(AZ별 1개, 웹 서버용)과 3개의 프라이빗 서브넷(AZ별 1개, 애플리케이션/데이터베이스 서버용). 공간을 균등하게 나누고 성장 여지를 남기는 CIDR 블록을 할당하세요.

<details>
<summary>정답 보기</summary>

`/16` VPC는 65,536개의 주소를 제공합니다. `/24` 서브넷(각 256개 주소)을 사용하면 충분한 여유가 있으며 단순성을 위한 일반적인 표준입니다.

| 서브넷 이름 | AZ | CIDR | 목적 |
|------------|-----|------|------|
| public-subnet-a | ap-northeast-2a | 10.100.1.0/24 | 웹 서버 (퍼블릭) |
| public-subnet-b | ap-northeast-2b | 10.100.2.0/24 | 웹 서버 (퍼블릭) |
| public-subnet-c | ap-northeast-2c | 10.100.3.0/24 | 웹 서버 (퍼블릭) |
| private-subnet-a | ap-northeast-2a | 10.100.11.0/24 | 앱/DB 서버 (프라이빗) |
| private-subnet-b | ap-northeast-2b | 10.100.12.0/24 | 앱/DB 서버 (프라이빗) |
| private-subnet-c | ap-northeast-2c | 10.100.13.0/24 | 앱/DB 서버 (프라이빗) |

**설계 원칙**:
- 퍼블릭 서브넷은 쉬운 식별을 위해 `.1.x`~`.3.x` 범위 사용
- 프라이빗 서브넷은 퍼블릭과 논리적으로 구분된 `.11.x`~`.13.x` 범위 사용
- 6개 서브넷은 `/16` 공간의 극히 일부만 사용 — 향후 서비스를 위한 수백 개의 추가 서브넷 가능
- AWS는 서브넷당 5개 IP를 예약(처음 4개 + 마지막 1개), `/24`당 251개의 사용 가능한 IP

</details>

### 연습 문제 2: 보안 그룹(Security Group) vs 네트워크 ACL(Network ACL)

웹 애플리케이션 구성:
- 퍼블릭 서브넷의 웹 서버 (80/443 인바운드)
- 프라이빗 서브넷의 애플리케이션 서버 (포트 8080, 웹 계층에서만)
- 프라이빗 서브넷의 데이터베이스 서버 (포트 5432, 앱 계층에서만)

애플리케이션 서버 보안 그룹의 인바운드 규칙을 작성하세요(소스, 포트, 프로토콜 설명). 그리고 보안 그룹과 NACL의 핵심 차이점 하나를 설명하세요.

<details>
<summary>정답 보기</summary>

**애플리케이션 서버 보안 그룹 — 인바운드 규칙**:

| 규칙 # | 유형 | 프로토콜 | 포트 | 소스 | 설명 |
|--------|------|--------|------|------|------|
| 1 | Custom TCP | TCP | 8080 | 웹 서버 SG ID | 웹 서버에서의 트래픽만 허용 |
| 2 | SSH | TCP | 22 | 배스천(bastion) SG ID 또는 관리자 CIDR | 배스천 호스트에서 관리자 SSH 허용 |

**AWS CLI 예시**:
```bash
# 웹 서버 보안 그룹에서 포트 8080 허용
aws ec2 authorize-security-group-ingress \
    --group-id sg-app-server \
    --protocol tcp \
    --port 8080 \
    --source-group sg-web-server
```

**보안 그룹과 NACL의 핵심 차이점**:

| | 보안 그룹(Security Group) | 네트워크 ACL(Network ACL) |
|--|--------------------------|--------------------------|
| **상태** | **스테이트풀(Stateful)** — 인바운드 트래픽이 허용되면 반환 트래픽이 자동으로 허용됨 | **스테이트리스(Stateless)** — 양방향 트래픽을 인바운드와 아웃바운드 규칙 모두에서 명시적으로 허용해야 함 |
| **범위** | 개별 EC2 인스턴스/ENI에 적용 | 전체 서브넷에 적용 |
| **규칙** | 허용 규칙만 (명시적 거부 없음) | 허용 및 거부 규칙 (번호 순서대로 처리) |

**스테이트리스의 실질적 의미**: NACL에 인바운드 HTTP(포트 80)를 허용하는 규칙이 있다면, 응답 패킷을 위해 임시 포트(1024~65535)에 대한 아웃바운드 규칙도 추가해야 합니다. 보안 그룹은 이를 자동으로 처리합니다.

</details>

### 연습 문제 3: NAT 게이트웨이(NAT Gateway) 목적과 설정

프라이빗 서브넷에 웹 애플리케이션을 실행하는 EC2 인스턴스가 있습니다. 인스턴스가 시작 시 인터넷에서 소프트웨어 패키지를 다운로드해야 하지만, 인터넷에서 직접 접근 가능해서는 안 됩니다.

1. 프라이빗 서브넷 인스턴스의 아웃바운드 전용 인터넷 접근을 가능하게 하는 AWS 컴포넌트는 무엇입니까?
2. 어떤 라우팅 테이블 항목이 필요합니까?
3. 필요한 컴포넌트를 생성하고 라우팅을 설정하는 AWS CLI 명령어를 작성하세요.

<details>
<summary>정답 보기</summary>

1. **NAT 게이트웨이(NAT Gateway)** — 프라이빗 서브넷 인스턴스에 아웃바운드 전용 인터넷 연결을 제공합니다. 퍼블릭 서브넷에 위치하며 퍼블릭 IP(탄력적 IP, Elastic IP)를 가집니다. 프라이빗 인스턴스는 아웃바운드 인터넷 트래픽을 NAT 게이트웨이를 통해 라우팅하고, NAT 게이트웨이가 트래픽을 전달하고 응답을 처리하지만, 인터넷에서 프라이빗 인스턴스로 트래픽을 시작하는 것은 차단됩니다.

2. **프라이빗 서브넷의 라우팅 테이블에 필요한 항목**:
   - 대상(Destination): `0.0.0.0/0`
   - 타깃(Target): `nat-xxxxxxxx` (NAT 게이트웨이 ID)

3. **AWS CLI 명령어**:
```bash
# 1단계: NAT 게이트웨이를 위한 탄력적 IP 할당
aws ec2 allocate-address --domain vpc

# 2단계: 퍼블릭 서브넷에 NAT 게이트웨이 생성
# (1단계의 할당 ID로 eipalloc-xxx 교체)
aws ec2 create-nat-gateway \
    --subnet-id subnet-public-a \
    --allocation-id eipalloc-xxx \
    --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=nat-gw-a}]'

# 3단계: 프라이빗 라우팅 테이블에 NAT 게이트웨이를 가리키는 기본 라우트 추가
# (rtb-private과 nat-xxx를 실제 ID로 교체)
aws ec2 create-route \
    --route-table-id rtb-private \
    --destination-cidr-block 0.0.0.0/0 \
    --nat-gateway-id nat-xxx
```

**비용 참고**: NAT 게이트웨이는 무료가 아닙니다. 시간당 요금(~$0.045/시간)과 데이터 처리량 GB당 요금이 발생합니다. 비용에 민감한 개발 환경에서는 NAT 인스턴스(Instance)를 더 저렴한 대안으로 고려할 수 있습니다.

</details>

### 연습 문제 4: VPC 피어링(VPC Peering) 시나리오

회사 A는 `ap-northeast-2`에 VPC-A(`10.0.0.0/16`)를 가지고 있고, 회사 B는 동일 리전에 VPC-B(`172.16.0.0/16`)를 가지고 있습니다. 두 회사의 애플리케이션 서버가 프라이빗 IP를 통해 직접 통신해야 합니다.

1. VPC 피어링이란 무엇이며, 핵심 제한 사항은 무엇입니까?
2. 피어링 연결을 설정하는 데 필요한 단계를 설명하세요.

<details>
<summary>정답 보기</summary>

1. **VPC 피어링(VPC Peering)**은 두 VPC 간의 네트워킹 연결로, 마치 동일한 네트워크 내에 있는 것처럼 프라이빗 IPv4 또는 IPv6 주소를 사용하여 트래픽을 라우팅할 수 있습니다.

   **핵심 제한 사항**:
   - **전이적 라우팅 없음(No transitive routing)**: VPC-A가 VPC-B와 피어링되고 VPC-B가 VPC-C와 피어링되어도, VPC-A는 VPC-B를 통해 VPC-C에 도달할 수 없습니다. 각 쌍은 직접 피어링이 필요합니다.
   - **CIDR 블록 겹침 불가**: VPC-A(`10.0.0.0/16`)와 VPC-B(`172.16.0.0/16`)는 겹치지 않으므로 피어링 가능합니다. 둘 다 `10.0.0.0/16`을 사용하면 피어링이 불가합니다.
   - **기본적으로 단일 리전** (교차 리전 피어링은 지원되지만 지연 시간과 데이터 전송 비용이 추가됩니다).

2. **VPC 피어링 설정 단계**:
```bash
# 1단계: 피어링 요청 생성 (VPC-A에서 VPC-B로)
aws ec2 create-vpc-peering-connection \
    --vpc-id vpc-A \
    --peer-vpc-id vpc-B \
    --peer-region ap-northeast-2

# 2단계: 피어링 요청 수락 (VPC-B 소유자가 수행)
aws ec2 accept-vpc-peering-connection \
    --vpc-peering-connection-id pcx-xxx

# 3단계: 양쪽 라우팅 테이블 업데이트
# VPC-A의 라우팅 테이블: VPC-B의 CIDR을 피어링 연결로 라우팅
aws ec2 create-route \
    --route-table-id rtb-A \
    --destination-cidr-block 172.16.0.0/16 \
    --vpc-peering-connection-id pcx-xxx

# VPC-B의 라우팅 테이블: VPC-A의 CIDR을 라우팅
aws ec2 create-route \
    --route-table-id rtb-B \
    --destination-cidr-block 10.0.0.0/16 \
    --vpc-peering-connection-id pcx-xxx

# 4단계: 양쪽의 보안 그룹을 업데이트하여 피어 VPC의 CIDR 블록에서의 트래픽 허용
```

</details>

### 연습 문제 5: 3-계층 아키텍처 설계

두 AZ에 걸친 고가용성 3-계층 웹 애플리케이션의 완전한 VPC 아키텍처를 그리거나 설명하세요. 서브넷, 인터넷 게이트웨이(Internet Gateway), NAT 게이트웨이, 보안 그룹, 라우팅을 포함해야 합니다.

<details>
<summary>정답 보기</summary>

```
VPC: 10.0.0.0/16  (리전: ap-northeast-2)
│
├── 인터넷 게이트웨이(Internet Gateway) (VPC에 연결됨)
│
├── AZ-a (ap-northeast-2a)
│   ├── 퍼블릭 서브넷 10.0.1.0/24
│   │   ├── ALB(Application Load Balancer) 노드
│   │   ├── NAT 게이트웨이 (탄력적 IP 포함)
│   │   └── 라우팅 테이블: 0.0.0.0/0 → IGW
│   │
│   ├── 프라이빗 서브넷 (앱) 10.0.11.0/24
│   │   ├── EC2 앱 서버 (Auto Scaling Group)
│   │   └── 라우팅 테이블: 0.0.0.0/0 → NAT-GW-a
│   │
│   └── 프라이빗 서브넷 (DB) 10.0.21.0/24
│       ├── RDS Primary
│       └── 라우팅 테이블: 로컬만 (인터넷 라우트 없음)
│
└── AZ-b (ap-northeast-2b)
    ├── 퍼블릭 서브넷 10.0.2.0/24
    │   ├── ALB 노드
    │   └── NAT 게이트웨이 (탄력적 IP 포함) [중복]
    │
    ├── 프라이빗 서브넷 (앱) 10.0.12.0/24
    │   ├── EC2 앱 서버
    │   └── 라우팅 테이블: 0.0.0.0/0 → NAT-GW-b
    │
    └── 프라이빗 서브넷 (DB) 10.0.22.0/24
        ├── RDS 스탠바이(Standby) (Multi-AZ)
        └── 라우팅 테이블: 로컬만

보안 그룹:
- ALB-SG: 0.0.0.0/0에서 인바운드 80/443
- App-SG: ALB-SG에서만 인바운드 8080
- DB-SG: App-SG에서만 인바운드 5432
```

**설계 근거**:
- 각 계층은 별도 서브넷에 위치하여 고유한 보안 그룹 정책 적용 가능
- 두 개의 NAT 게이트웨이(AZ당 1개)로 한 AZ가 실패해도 두 AZ의 프라이빗 서브넷이 인터넷 접근 유지
- 데이터베이스 서브넷에는 인터넷 라우트 없음 — DB 인스턴스는 어떤 상황에서도 인터넷 트래픽을 시작하거나 받을 수 없음
- ALB가 두 퍼블릭 서브넷에 걸쳐 있어 두 AZ의 정상 앱 서버로 트래픽을 라우팅

</details>

---

## 참고 자료

- [AWS VPC Documentation](https://docs.aws.amazon.com/vpc/)
- [GCP VPC Documentation](https://cloud.google.com/vpc/docs)
- [AWS VPC Best Practices](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-best-practices.html)
- [Networking/](../Networking/) - 네트워크 기초 이론
