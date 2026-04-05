# 리전과 가용 영역

**이전**: [AWS & GCP 계정 설정](./02_AWS_GCP_Account_Setup.md) | **다음**: [가상 머신](./04_Virtual_Machines.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 리전(Region), 가용 영역(Availability Zone), 데이터센터의 계층 구조를 설명할 수 있습니다
2. AWS와 GCP의 글로벌 인프라 모델을 비교할 수 있습니다
3. 리전 선택에 영향을 미치는 요소(지연 시간, 규정 준수, 비용, 서비스 가용성)를 파악할 수 있습니다
4. 고가용성을 위한 멀티 AZ(Multi-AZ) 배포를 설계할 수 있습니다
5. 리전 범위, 존 범위, 글로벌 범위 클라우드 서비스를 구분할 수 있습니다
6. 콘텐츠 전송에서 엣지 로케이션(Edge Location)의 역할을 설명할 수 있습니다

---

모든 클라우드 리소스는 지구 어딘가에 있는 물리적 데이터센터에서 실행됩니다. 올바른 리전을 선택하면 사용자 지연 시간, 데이터 거주(Data Residency) 법규 준수, 서비스 가용성, 심지어 비용에도 영향을 미칩니다. 신뢰성 높고 성능 좋은 아키텍처를 설계하려면 클라우드 제공자가 글로벌 인프라를 어떻게 구성하는지 이해하는 것이 필수입니다.

## 1. 글로벌 인프라 개요

클라우드 제공자는 전 세계에 분산된 데이터센터를 통해 서비스를 제공합니다.

### 1.1 인프라 계층 구조

```
글로벌 네트워크
├── 리전 (Region)
│   ├── 가용 영역 (Availability Zone / Zone)
│   │   └── 데이터센터
│   ├── 가용 영역
│   │   └── 데이터센터
│   └── 가용 영역
│       └── 데이터센터
├── 리전
│   └── ...
└── 엣지 로케이션 (CDN, DNS)
```

### 1.2 AWS vs GCP 용어 비교

| 개념 | AWS | GCP |
|------|-----|-----|
| 지리적 영역 | Region | Region |
| 독립 데이터센터 | Availability Zone (AZ) | Zone |
| 로컬 서비스 | Local Zones, Wavelength | - |
| CDN 엣지 | Edge Locations | Edge PoPs |
| 프라이빗 연결 | Direct Connect | Cloud Interconnect |

---

## 2. 리전 (Region)

### 2.1 정의

리전은 지리적으로 분리된 클라우드 인프라 영역입니다.

**특징:**
- 각 리전은 독립적으로 운영
- 리전 간 데이터 복제는 명시적 설정 필요
- 대부분의 서비스는 리전 단위로 제공

### 2.2 AWS 주요 리전

| 리전 코드 | 위치 | 한국에서 권장 |
|----------|------|--------------|
| ap-northeast-2 | 서울 | ✅ 가장 권장 |
| ap-northeast-1 | 도쿄 | ✅ 차선책 |
| ap-northeast-3 | 오사카 | 선택적 |
| ap-southeast-1 | 싱가포르 | 선택적 |
| us-east-1 | 버지니아 북부 | 글로벌 서비스 |
| us-west-2 | 오레곤 | 비용 최적화 |
| eu-west-1 | 아일랜드 | 유럽 서비스 |

```bash
# 현재 리전 확인
aws configure get region

# 리전 설정
aws configure set region ap-northeast-2

# 사용 가능한 리전 목록
aws ec2 describe-regions --output table
```

### 2.3 GCP 주요 리전

| 리전 코드 | 위치 | 한국에서 권장 |
|----------|------|--------------|
| asia-northeast3 | 서울 | ✅ 가장 권장 |
| asia-northeast1 | 도쿄 | ✅ 차선책 |
| asia-northeast2 | 오사카 | 선택적 |
| asia-southeast1 | 싱가포르 | 선택적 |
| us-central1 | 아이오와 | 무료 티어 |
| us-east1 | 사우스캐롤라이나 | 무료 티어 |
| europe-west1 | 벨기에 | 유럽 서비스 |

```bash
# 현재 리전 확인
gcloud config get-value compute/region

# 리전 설정
gcloud config set compute/region asia-northeast3

# 사용 가능한 리전 목록
gcloud compute regions list
```

---

## 3. 가용 영역 (Availability Zone / Zone)

### 3.1 정의

가용 영역은 리전 내의 독립적인 데이터센터 그룹입니다.

**특징:**
- 물리적으로 분리된 위치
- 독립적인 전력, 냉각, 네트워크
- 저지연 고속 네트워크로 연결
- 한 AZ 장애가 다른 AZ에 영향 없음

### 3.2 AWS 가용 영역

```
서울 리전 (ap-northeast-2)
├── ap-northeast-2a
├── ap-northeast-2b
├── ap-northeast-2c
└── ap-northeast-2d
```

**AZ 명명 규칙:**
- `{리전코드}{영역문자}` 형식
- 예: `ap-northeast-2a`, `us-east-1b`

```bash
# 가용 영역 목록 확인
aws ec2 describe-availability-zones --region ap-northeast-2

# 출력 예시
{
    "AvailabilityZones": [
        {
            "ZoneName": "ap-northeast-2a",
            "State": "available",
            "ZoneType": "availability-zone"
        },
        ...
    ]
}
```

### 3.3 GCP Zone

```
서울 리전 (asia-northeast3)
├── asia-northeast3-a
├── asia-northeast3-b
└── asia-northeast3-c
```

**Zone 명명 규칙:**
- `{리전코드}-{영역문자}` 형식
- 예: `asia-northeast3-a`, `us-central1-f`

```bash
# Zone 목록 확인
gcloud compute zones list --filter="region:asia-northeast3"

# 출력 예시
NAME                 REGION           STATUS
asia-northeast3-a    asia-northeast3  UP
asia-northeast3-b    asia-northeast3  UP
asia-northeast3-c    asia-northeast3  UP
```

---

## 4. 멀티 AZ 아키텍처

### 4.1 고가용성을 위한 설계

```
┌────────────────────────────────────────────────────────────┐
│                     리전 (Region)                          │
│  ┌──────────────────┐    ┌──────────────────┐             │
│  │    AZ-a          │    │    AZ-b          │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │   Web-1    │  │    │  │   Web-2    │  │             │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │   App-1    │  │    │  │   App-2    │  │             │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │  DB-Primary │ │───▶│  │ DB-Standby │  │  (동기 복제) │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  └──────────────────┘    └──────────────────┘             │
│                                                            │
│  ┌──────────────────────────────────────────┐             │
│  │           Load Balancer (리전 범위)       │             │
│  └──────────────────────────────────────────┘             │
└────────────────────────────────────────────────────────────┘
```

### 4.2 서비스별 Multi-AZ 옵션

**AWS:**

| 서비스 | Multi-AZ 방식 |
|--------|--------------|
| EC2 | Auto Scaling Group으로 분산 |
| RDS | Multi-AZ 옵션 활성화 |
| ElastiCache | 복제본 다른 AZ 배치 |
| ELB | 자동 Multi-AZ |
| S3 | 자동 Multi-AZ 복제 |

**GCP:**

| 서비스 | Multi-Zone 방식 |
|--------|----------------|
| Compute Engine | Instance Group으로 분산 |
| Cloud SQL | 고가용성 옵션 활성화 |
| Memorystore | 복제본 다른 Zone 배치 |
| Cloud Load Balancing | 자동 Multi-Zone |
| Cloud Storage | Regional 클래스 사용 |

---

## 5. 리전 선택 기준

### 5.1 주요 고려 사항

| 기준 | 설명 | 권장 |
|------|------|------|
| **지연 시간** | 사용자와의 물리적 거리 | 사용자 근처 리전 |
| **규정 준수** | 데이터 거주 요구사항 | 법적 요구사항 확인 |
| **서비스 가용성** | 모든 서비스가 모든 리전에 없음 | 필요 서비스 확인 |
| **비용** | 리전별 가격 차이 | 비용 비교 |
| **재해 복구** | DR 사이트 거리 | 충분히 먼 리전 |

### 5.2 지연 시간 테스트

**AWS 지연 시간 측정:**
```bash
# CloudPing 사이트 활용
# https://www.cloudping.info/

# 또는 직접 ping 테스트
ping ec2.ap-northeast-2.amazonaws.com
ping ec2.ap-northeast-1.amazonaws.com
```

**GCP 지연 시간 측정:**
```bash
# GCP Ping 테스트 사이트
# https://gcping.com/

# 또는 직접 측정
ping asia-northeast3-run.googleapis.com
```

### 5.3 서비스 가용성 확인

**AWS:**
- https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/

**GCP:**
- https://cloud.google.com/about/locations

### 5.4 비용 비교 (EC2/Compute Engine 예시)

| 인스턴스 타입 | 서울 (AWS) | 버지니아 (AWS) | 서울 (GCP) | 아이오와 (GCP) |
|--------------|-----------|---------------|-----------|---------------|
| 범용 2vCPU/8GB | ~$0.10/시간 | ~$0.08/시간 | ~$0.09/시간 | ~$0.07/시간 |

*가격은 변동될 수 있으므로 공식 가격표 확인 필요*

---

## 6. 글로벌/리전/존 서비스

### 6.1 AWS 서비스 범위

| 범위 | 서비스 예시 |
|------|-----------|
| **글로벌** | IAM, Route 53, CloudFront, WAF |
| **리전** | VPC, S3, Lambda, RDS, EC2 (AMI) |
| **가용 영역** | EC2 인스턴스, EBS 볼륨, 서브넷 |

### 6.2 GCP 서비스 범위

| 범위 | 서비스 예시 |
|------|-----------|
| **글로벌** | Cloud IAM, Cloud DNS, Cloud CDN, VPC (네트워크) |
| **리전** | Cloud Storage (Regional), Cloud SQL, Cloud Run |
| **존** | Compute Engine, Persistent Disk |

**GCP VPC 특이점:**
- GCP의 VPC는 **글로벌** 리소스 (AWS VPC는 리전 단위)
- 서브넷은 리전 범위

```
AWS VPC vs GCP VPC

AWS:
├── VPC (리전 범위) ─── us-east-1
│   ├── Subnet-a (AZ 범위) ─── us-east-1a
│   └── Subnet-b (AZ 범위) ─── us-east-1b
└── VPC (별도 리전) ─── ap-northeast-2
    └── Subnet-a ─── ap-northeast-2a

GCP:
└── VPC (글로벌)
    ├── Subnet-us (리전 범위) ─── us-central1
    ├── Subnet-asia (리전 범위) ─── asia-northeast3
    └── Subnet-eu (리전 범위) ─── europe-west1
```

---

## 7. 엣지 로케이션

### 7.1 CDN 엣지

**AWS CloudFront:**
- 400+ 엣지 로케이션
- 정적 콘텐츠 캐싱
- DDoS 보호 (AWS Shield)

**GCP Cloud CDN:**
- Google의 글로벌 엣지 네트워크 활용
- 자동 SSL/TLS
- Cloud Armor 통합

### 7.2 DNS 엣지

**AWS Route 53:**
- 글로벌 Anycast DNS
- 지연 시간 기반 라우팅
- 지리적 라우팅

**GCP Cloud DNS:**
- 글로벌 Anycast
- 100% 가용성 SLA
- DNSSEC 지원

---

## 8. 재해 복구 전략

### 8.1 DR 패턴

| 패턴 | RTO | RPO | 비용 | 설명 |
|------|-----|-----|------|------|
| **Backup & Restore** | 시간~일 | 시간~일 | 낮음 | 백업만 다른 리전에 저장 |
| **Pilot Light** | 분~시간 | 분~시간 | 중간 | 핵심 시스템만 대기 |
| **Warm Standby** | 분 | 분 | 높음 | 축소된 환경 상시 운영 |
| **Active-Active** | 초 | 거의 0 | 매우 높음 | 모든 리전 동시 운영 |

### 8.2 크로스 리전 복제

**AWS S3 크로스 리전 복제:**
```bash
# S3 버킷 복제 설정
aws s3api put-bucket-replication \
    --bucket source-bucket \
    --replication-configuration '{
        "Role": "arn:aws:iam::account-id:role/replication-role",
        "Rules": [{
            "Status": "Enabled",
            "Destination": {
                "Bucket": "arn:aws:s3:::destination-bucket"
            }
        }]
    }'
```

**GCP Cloud Storage 복제:**
```bash
# Dual-region 또는 Multi-region 버킷 사용
gsutil mb -l asia gs://my-multi-region-bucket

# 또는 Storage Transfer Service로 복제
gcloud transfer jobs create \
    gs://source-bucket gs://destination-bucket
```

---

## 9. 실습: 리전/AZ 정보 조회

### 9.1 AWS CLI 실습

```bash
# 1. 모든 리전 목록
aws ec2 describe-regions --query 'Regions[*].RegionName' --output text

# 2. 서울 리전의 AZ 목록
aws ec2 describe-availability-zones \
    --region ap-northeast-2 \
    --query 'AvailabilityZones[*].[ZoneName,State]' \
    --output table

# 3. 특정 서비스의 리전별 가용성 확인 (SSM 파라미터)
aws ssm get-parameters-by-path \
    --path /aws/service/global-infrastructure/regions \
    --query 'Parameters[*].Name'
```

### 9.2 GCP gcloud 실습

```bash
# 1. 모든 리전 목록
gcloud compute regions list --format="value(name)"

# 2. 서울 리전의 Zone 목록
gcloud compute zones list \
    --filter="region:asia-northeast3" \
    --format="table(name,status)"

# 3. 특정 리전의 머신 타입 확인
gcloud compute machine-types list \
    --filter="zone:asia-northeast3-a" \
    --limit=10
```

---

## 10. 다음 단계

- [04_Virtual_Machines.md](./04_Virtual_Machines.md) - 가상 머신 생성 및 관리
- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC 네트워킹

---

## 연습 문제

### 연습 문제 1: 인프라 계층(Hierarchy) 식별

각 AWS 리소스에 대해 범위(글로벌, 리전, 가용 영역)를 식별하세요:

1. `ap-northeast-2a`에서 실행 중인 EC2 인스턴스
2. `my-data-bucket`이라는 이름의 S3 버킷
3. `EC2ReadOnlyRole`이라는 이름의 IAM 역할
4. RDS Multi-AZ 인스턴스
5. `us-east-1b`의 EBS 볼륨
6. Route 53 호스팅 존(hosted zone)

<details>
<summary>정답 보기</summary>

1. **가용 영역(Availability Zone)** — EC2 인스턴스는 특정 AZ에 묶여 있습니다. 해당 인스턴스는 `ap-northeast-2a`에서만 실행됩니다.
2. **리전(Regional)** — S3 버킷은 리전 범위입니다. AWS가 리전 내 여러 AZ에 걸쳐 데이터를 자동으로 복제하지만, 버킷은 특정 리전에 속합니다.
3. **글로벌(Global)** — IAM은 글로벌 서비스입니다. IAM 역할, 사용자, 정책은 계정의 모든 리전에 적용됩니다.
4. **리전(Regional)** — RDS Multi-AZ 인스턴스는 리전 리소스입니다. AWS가 동일 리전 내 다른 AZ에 스탠바이(standby) 복제본을 자동으로 관리합니다.
5. **가용 영역(Availability Zone)** — EBS 볼륨은 생성된 특정 AZ에 묶여 있습니다. 다른 AZ의 인스턴스에 직접 연결할 수 없습니다.
6. **글로벌(Global)** — Route 53는 글로벌 서비스입니다. 호스팅 존과 DNS 레코드는 전 세계에서 접근 가능합니다.

</details>

### 연습 문제 2: 리전 선택 결정

한국의 의료 스타트업이 환자 기록 관리 시스템을 구축하고 있습니다. 한국의 개인정보 보호법(PIPA)을 준수해야 하며, 이 법은 환자 데이터가 한국 내에 있어야 함을 요구합니다. 또한 소수의 EU 고객도 있습니다.

1. 주 애플리케이션과 데이터베이스의 기본 AWS 리전은 어디로 해야 합니까?
2. 시스템의 일부에 `us-east-1`을 사용할 수 있습니까? 가능하다면 어떤 부분에?
3. EU 고객에 대한 리전 고려 사항은 무엇입니까?

<details>
<summary>정답 보기</summary>

1. **`ap-northeast-2` (서울)** — PIPA에 따라 환자 데이터는 한국 내에 있어야 합니다. 서울은 한국 땅에 있는 유일한 AWS 리전이므로, 주 애플리케이션과 모든 규제 대상 환자 데이터 스토리지(RDS, 개인 건강 정보가 포함된 S3 등)에 필수적입니다.

2. **네, 글로벌/비규제 서비스에는 가능합니다** — `us-east-1`은 일부 글로벌 서비스 엔드포인트를 호스팅합니다(예: IAM, CloudFront 배포). 익명화된 분석, 내부 도구, 정적 마케팅 자산 등 개인 건강 정보(PHI, Protected Health Information)가 저장·처리되지 않는 경우에도 허용됩니다.

3. **EU 데이터 상주 요건** — EU 고객의 데이터에는 GDPR이 적용될 수 있으며, EU 환자 데이터가 EU를 벗어나지 않아야 할 수 있습니다. 이 경우 `eu-west-1`(아일랜드) 또는 `eu-central-1`(프랑크푸르트) 리전 배포가 필요합니다. 결과적으로 한국 환자는 `ap-northeast-2`, EU 환자는 EU 리전을 사용하는 멀티 리전 아키텍처가 됩니다.

</details>

### 연습 문제 3: Multi-AZ 아키텍처 설계

서울 리전(`ap-northeast-2`)에서 고가용성 웹 애플리케이션을 AWS에 설계하고 있습니다. 애플리케이션에는 로드 밸런서(load balancer), 애플리케이션 서버, 자동 장애 조치가 있는 관계형 데이터베이스가 필요합니다.

어떤 컴포넌트가 어떤 AZ에 위치해야 하는지와 그 이유를 설명하는 아키텍처를 그리거나 설명하세요.

<details>
<summary>정답 보기</summary>

```
서울 리전 (ap-northeast-2)
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  [Application Load Balancer]  ← 리전 범위, AZ 전체에 걸침  │
│             │                                              │
│    ┌────────┴────────┐                                     │
│    ▼                 ▼                                     │
│  AZ-a               AZ-b                                   │
│  ┌──────────────┐   ┌──────────────┐                       │
│  │ EC2 App-1    │   │ EC2 App-2    │  ← Auto Scaling Group  │
│  └──────────────┘   └──────────────┘                       │
│  ┌──────────────┐   ┌──────────────┐                       │
│  │ RDS Primary  │──▶│ RDS Standby  │  ← Multi-AZ RDS       │
│  │ (활성)       │   │ (대기)       │                       │
│  └──────────────┘   └──────────────┘                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**컴포넌트 설명**:
- **ALB** — 두 AZ의 정상 인스턴스로 트래픽을 자동으로 라우팅합니다. AZ-a가 장애 시 AZ-b 인스턴스로만 라우팅합니다.
- **Auto Scaling Group의 EC2** — `ap-northeast-2a`와 `ap-northeast-2b`에 분산됩니다. 각 AZ에 최소 하나의 인스턴스가 있습니다. 한 AZ가 장애 시 ASG는 살아남은 AZ에 대체 인스턴스를 시작합니다.
- **RDS Multi-AZ** — AZ-a에 Primary, AZ-b에 동기식 스탠바이 복제본. AZ-a가 장애 시 AWS가 자동으로 스탠바이를 승격합니다(일반적으로 2분 미만). 수동 개입이 불필요합니다.

**두 개의 AZ를 사용하는 이유?** AZ는 독립적인 전원, 냉각, 네트워킹을 가지므로 한 AZ의 장애(하드웨어 고장, 정전, 유지보수)가 다른 AZ에 영향을 미치지 않습니다.

</details>

### 연습 문제 4: AWS vs GCP VPC 범위 차이

AWS와 GCP가 VPC(Virtual Private Cloud) 범위를 정의하는 근본적인 차이를 설명하고, GCP 방식이 운영적으로 더 단순한 시나리오를 설명하세요.

<details>
<summary>정답 보기</summary>

**AWS VPC**: 리전 범위(Regional scope). VPC는 단일 리전으로 제한됩니다. `us-east-1`과 `ap-northeast-2`의 리소스가 프라이빗으로 통신하려면 각 리전에 별도의 VPC를 생성하고 VPC 피어링(VPC Peering) 또는 Transit Gateway로 연결해야 합니다.

**GCP VPC**: 글로벌 범위(Global scope). 단일 VPC가 모든 리전에 걸쳐 있습니다. VPC 내 서브넷(subnet)은 리전 범위이지만, 모두 동일한 라우팅 테이블(routing table)과 프라이빗 IP 공간을 공유합니다. `us-central1`의 VM과 `asia-northeast3`의 VM은 추가 설정 없이 동일 VPC 내에서 프라이빗 IP로 통신할 수 있습니다.

**GCP가 더 단순한 시나리오**: 지연 시간(latency) 최적화를 위해 여러 리전에 서비스를 배포하는 글로벌 마이크로서비스(microservices) 애플리케이션. GCP에서는 모든 서비스가 하나의 VPC를 공유합니다. `asia-northeast3`의 데이터베이스에 `us-central1`의 API 서버가 피어링 설정 없이 프라이빗 IP로 접근할 수 있습니다. AWS에서는 리전 간 VPC 피어링(또는 Transit Gateway)을 설정하고 CIDR 범위가 겹치지 않도록 관리해야 합니다. GCP의 단일 글로벌 VPC는 멀티 리전 아키텍처의 네트워킹 구성 복잡도를 크게 줄여줍니다.

</details>

### 연습 문제 5: 인프라 탐색을 위한 CLI 명령어

다음 작업을 수행하는 CLI 명령어를 작성하세요:

1. (AWS) 도쿄 리전(`ap-northeast-1`)의 모든 가용 영역과 현재 상태를 나열합니다.
2. (GCP) 서울 리전(`asia-northeast3`)에서 사용 가능한 모든 존(zone)을 나열합니다.
3. (AWS) CLI 기본 리전을 서울로 설정합니다.

<details>
<summary>정답 보기</summary>

1. **AWS — 도쿄 AZ 목록 조회**:
```bash
aws ec2 describe-availability-zones \
    --region ap-northeast-1 \
    --query 'AvailabilityZones[*].[ZoneName,State]' \
    --output table
```

2. **GCP — 서울 존 목록 조회**:
```bash
gcloud compute zones list \
    --filter="region:asia-northeast3" \
    --format="table(name,status)"
```

3. **AWS — 기본 리전을 서울로 설정**:
```bash
aws configure set region ap-northeast-2
```
현재 셸 세션에서만 설정하려면:
```bash
export AWS_DEFAULT_REGION=ap-northeast-2
```

</details>

---

## 참고 자료

- [AWS Global Infrastructure](https://aws.amazon.com/about-aws/global-infrastructure/)
- [GCP Locations](https://cloud.google.com/about/locations)
- [AWS Regions and Availability Zones](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html)
- [GCP Regions and Zones](https://cloud.google.com/compute/docs/regions-zones)
