# 클라우드 컴퓨팅 학습 가이드 (AWS & GCP)

## 소개

이 폴더는 AWS(Amazon Web Services)와 GCP(Google Cloud Platform)를 병렬 비교하며 학습할 수 있는 클라우드 컴퓨팅 자료를 담고 있습니다. 두 플랫폼의 핵심 서비스를 대응시켜 설명하므로, 한 플랫폼을 익히면 다른 플랫폼으로 쉽게 전환할 수 있습니다.

**대상 독자**: 클라우드 입문자 ~ 중급자 (실무 기초)

---

## 파일 목록

| 파일명 | 주제 | 난이도 | AWS 서비스 | GCP 서비스 |
|--------|------|--------|-----------|-----------|
| [01_Cloud_Computing_Overview.md](./01_Cloud_Computing_Overview.md) | 클라우드 개요, IaaS/PaaS/SaaS | ⭐ | - | - |
| [02_AWS_GCP_Account_Setup.md](./02_AWS_GCP_Account_Setup.md) | 계정 생성, 콘솔 탐색, MFA | ⭐ | Console | Console |
| [03_Regions_Availability_Zones.md](./03_Regions_Availability_Zones.md) | 리전, 가용 영역, 글로벌 인프라 | ⭐⭐ | Regions/AZs | Regions/Zones |
| [04_Virtual_Machines.md](./04_Virtual_Machines.md) | 가상 머신 생성, SSH 접속 | ⭐⭐ | EC2 | Compute Engine |
| [05_Serverless_Functions.md](./05_Serverless_Functions.md) | 서버리스 함수, 트리거 | ⭐⭐⭐ | Lambda | Cloud Functions |
| [06_Container_Services.md](./06_Container_Services.md) | 컨테이너, Kubernetes | ⭐⭐⭐ | ECS, EKS, Fargate | GKE, Cloud Run |
| [07_Object_Storage.md](./07_Object_Storage.md) | 객체 스토리지, 정적 호스팅 | ⭐⭐ | S3 | Cloud Storage |
| [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) | 블록/파일 스토리지, 스냅샷 | ⭐⭐⭐ | EBS, EFS | PD, Filestore |
| [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) | VPC, 서브넷, NAT | ⭐⭐⭐ | VPC | VPC |
| [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) | 로드밸런서, CDN | ⭐⭐⭐ | ELB, CloudFront | Cloud LB, CDN |
| [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) | 관리형 RDB, 복제 | ⭐⭐⭐ | RDS, Aurora | Cloud SQL |
| [12_NoSQL_Databases.md](./12_NoSQL_Databases.md) | NoSQL, 캐시 서비스 | ⭐⭐⭐ | DynamoDB, ElastiCache | Firestore, Memorystore |
| [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) | IAM 사용자, 역할, 정책 | ⭐⭐⭐ | IAM | IAM |
| [14_Security_Services.md](./14_Security_Services.md) | 보안그룹, KMS, 비밀관리 | ⭐⭐⭐⭐ | SG, KMS, Secrets Manager | Firewall, KMS, Secret Manager |
| [15_CLI_and_SDK.md](./15_CLI_and_SDK.md) | CLI 설치, SDK 사용 | ⭐⭐ | AWS CLI, boto3 | gcloud, google-cloud |
| [16_Infrastructure_as_Code.md](./16_Infrastructure_as_Code.md) | Terraform 기초, 모듈 | ⭐⭐⭐⭐ | Terraform | Terraform |
| [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) | 모니터링, 로그, 비용 관리 | ⭐⭐⭐ | CloudWatch, Cost Explorer | Cloud Monitoring, Billing |

**총 레슨**: 17개

---

## 학습 순서

### Phase 1: 클라우드 기초 (01-03)
1. **클라우드 개념 이해**: 01_Cloud_Computing_Overview
2. **계정 설정**: 02_AWS_GCP_Account_Setup
3. **인프라 구조 이해**: 03_Regions_Availability_Zones

### Phase 2: 컴퓨팅 서비스 (04-06)
4. **가상 머신**: 04_Virtual_Machines (EC2 / Compute Engine)
5. **서버리스**: 05_Serverless_Functions (Lambda / Cloud Functions)
6. **컨테이너**: 06_Container_Services (EKS / GKE)

### Phase 3: 스토리지 (07-08)
7. **객체 스토리지**: 07_Object_Storage (S3 / Cloud Storage)
8. **블록/파일 스토리지**: 08_Block_and_File_Storage (EBS / Persistent Disk)

### Phase 4: 네트워킹 (09-10)
9. **VPC 설계**: 09_Virtual_Private_Cloud
10. **로드밸런싱 & CDN**: 10_Load_Balancing_CDN

### Phase 5: 데이터베이스 (11-12)
11. **관계형 DB**: 11_Managed_Relational_DB (RDS / Cloud SQL)
12. **NoSQL DB**: 12_NoSQL_Databases (DynamoDB / Firestore)

### Phase 6: 보안 (13-14)
13. **IAM**: 13_Identity_Access_Management
14. **보안 서비스**: 14_Security_Services

### Phase 7: DevOps & IaC (15-16)
15. **CLI/SDK**: 15_CLI_and_SDK
16. **Infrastructure as Code**: 16_Infrastructure_as_Code (Terraform)

### Phase 8: 운영 (17)
17. **모니터링 & 비용**: 17_Monitoring_Logging_Cost

---

## AWS vs GCP 서비스 매핑 요약

| 카테고리 | AWS | GCP |
|----------|-----|-----|
| **컴퓨팅** | EC2, Lambda, ECS, EKS | Compute Engine, Cloud Functions, Cloud Run, GKE |
| **스토리지** | S3, EBS, EFS | Cloud Storage, Persistent Disk, Filestore |
| **네트워킹** | VPC, ELB, CloudFront | VPC, Cloud Load Balancing, Cloud CDN |
| **데이터베이스** | RDS, Aurora, DynamoDB | Cloud SQL, Spanner, Firestore |
| **보안** | IAM, KMS, Secrets Manager | IAM, Cloud KMS, Secret Manager |
| **모니터링** | CloudWatch, X-Ray | Cloud Monitoring, Cloud Trace |
| **IaC** | CloudFormation | Deployment Manager |
| **CLI** | AWS CLI | gcloud CLI |

---

## 실습 환경 준비

### 필수 도구

```bash
# AWS CLI 설치
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"  # macOS
# 또는: pip install awscli

# GCP gcloud CLI 설치
curl https://sdk.cloud.google.com | bash

# Terraform 설치
brew install terraform  # macOS
# 또는: https://www.terraform.io/downloads
```

### 무료 티어 활용

| 플랫폼 | 무료 티어 |
|--------|----------|
| **AWS** | 12개월 무료 (t2.micro EC2, 5GB S3 등) |
| **GCP** | $300 크레딧 (90일), Always Free 서비스 |

---

## 관련 자료

- [AWS 공식 문서](https://docs.aws.amazon.com/)
- [GCP 공식 문서](https://cloud.google.com/docs)
- [Terraform 공식 문서](https://www.terraform.io/docs)
- [AWS vs GCP 비교 가이드](https://cloud.google.com/docs/compare/aws)

### 관련 폴더

- [Docker/](../Docker/) - 컨테이너 기초, Kubernetes 입문
- [Networking/](../Networking/) - 네트워크 기초 이론
- [Linux/](../Linux/) - 서버 관리 기초
