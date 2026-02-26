# IAM (Identity and Access Management)

**이전**: [NoSQL 데이터베이스](./12_NoSQL_Databases.md) | **다음**: [보안 서비스](./14_Security_Services.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. IAM의 목적과 최소 권한 원칙(principle of least privilege)을 설명할 수 있습니다
2. 범위, 정책 모델, 역할 바인딩(role-binding) 메커니즘 측면에서 AWS IAM과 GCP IAM을 비교할 수 있습니다
3. 적절한 권한 정책을 가진 사용자, 그룹, 역할을 생성할 수 있습니다
4. JSON 정책 문서(AWS) 또는 역할 바인딩(GCP)을 사용하여 IAM 정책을 작성하고 연결할 수 있습니다
5. 머신 간 인증(machine-to-machine authentication)을 위한 서비스 계정(service account)을 구성할 수 있습니다
6. IAM 역할 위임(role assumption)을 사용하여 교차 계정 접근(cross-account access)을 구현할 수 있습니다
7. IAM 구성을 감사하여 과도하게 권한이 부여된 주체(over-permissioned principal)를 식별할 수 있습니다

---

IAM(Identity and Access Management)은 클라우드 보안의 핵심입니다. 모든 API 호출, 모든 리소스 접근, 모든 서비스 상호작용은 IAM 정책에 의해 제어됩니다. 단 하나의 잘못 구성된 권한이 민감한 데이터를 노출시키거나 공격자에게 발판을 제공할 수 있습니다. IAM 숙달은 선택 사항이 아닙니다 -- 이는 모든 클라우드 환경에서 첫 번째이자 가장 중요한 보안 통제 수단입니다.

## 1. IAM 개요

### 1.1 IAM이란?

IAM은 클라우드 리소스에 대한 접근을 안전하게 제어하는 서비스입니다.

**핵심 질문:**
- **누가 (Who)**: 사용자, 그룹, 역할
- **무엇을 (What)**: 리소스
- **어떻게 (How)**: 권한 (허용/거부)

### 1.2 AWS vs GCP IAM 비교

| 항목 | AWS IAM | GCP IAM |
|------|---------|---------|
| 범위 | 계정 수준 | 조직/프로젝트 수준 |
| 정책 부착 | 사용자/그룹/역할에 | 리소스에 |
| 역할 | 역할을 맡음 (AssumeRole) | 역할 바인딩 |
| 서비스 계정 | 역할 + 인스턴스 프로파일 | 서비스 계정 |

---

## 2. AWS IAM

### 2.1 핵심 개념

```
┌─────────────────────────────────────────────────────────────┐
│  AWS 계정                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  IAM                                                    ││
│  │  ┌───────────────┐  ┌───────────────┐                   ││
│  │  │    사용자     │  │     그룹      │                   ││
│  │  │  (Users)      │  │  (Groups)     │                   ││
│  │  └───────────────┘  └───────────────┘                   ││
│  │         ↓                  ↓                            ││
│  │  ┌─────────────────────────────────────────────┐        ││
│  │  │              정책 (Policies)                │        ││
│  │  │  { "Effect": "Allow",                       │        ││
│  │  │    "Action": "s3:*",                        │        ││
│  │  │    "Resource": "*" }                        │        ││
│  │  └─────────────────────────────────────────────┘        ││
│  │                     ↓                                   ││
│  │  ┌───────────────────────────────────────────────────┐  ││
│  │  │              역할 (Roles)                         │  ││
│  │  │  - EC2 인스턴스 역할                              │  ││
│  │  │  - Lambda 실행 역할                               │  ││
│  │  │  - 교차 계정 역할                                 │  ││
│  │  └───────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 2.2 사용자 및 그룹

```bash
# 사용자 생성
aws iam create-user --user-name john

# 로그인 비밀번호 설정
aws iam create-login-profile \
    --user-name john \
    --password 'TempPassword123!' \
    --password-reset-required

# 액세스 키 생성 (프로그래밍 접근)
aws iam create-access-key --user-name john

# 그룹 생성
aws iam create-group --group-name Developers

# 그룹에 사용자 추가
aws iam add-user-to-group --group-name Developers --user-name john

# 그룹 멤버 확인
aws iam get-group --group-name Developers
```

### 2.3 정책 (Policies)

**정책 구조:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3Read",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ],
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "203.0.113.0/24"
                }
            }
        }
    ]
}
```

```bash
# 관리형 정책 연결
aws iam attach-user-policy \
    --user-name john \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# 커스텀 정책 생성
aws iam create-policy \
    --policy-name MyS3Policy \
    --policy-document file://policy.json

# 그룹에 정책 연결
aws iam attach-group-policy \
    --group-name Developers \
    --policy-arn arn:aws:iam::123456789012:policy/MyS3Policy

# 인라인 정책 추가
aws iam put-user-policy \
    --user-name john \
    --policy-name InlinePolicy \
    --policy-document file://inline-policy.json
```

### 2.4 역할 (Roles)

**EC2 인스턴스 역할:**
```bash
# 신뢰 정책 (누가 역할을 맡을 수 있는지)
cat > trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# 역할 생성
aws iam create-role \
    --role-name EC2-S3-Access \
    --assume-role-policy-document file://trust-policy.json

# 정책 연결
aws iam attach-role-policy \
    --role-name EC2-S3-Access \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# 인스턴스 프로파일 생성 및 역할 추가
aws iam create-instance-profile --instance-profile-name EC2-S3-Profile
aws iam add-role-to-instance-profile \
    --instance-profile-name EC2-S3-Profile \
    --role-name EC2-S3-Access

# EC2에 인스턴스 프로파일 연결
aws ec2 associate-iam-instance-profile \
    --instance-id i-1234567890abcdef0 \
    --iam-instance-profile Name=EC2-S3-Profile
```

**교차 계정 역할:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::OTHER_ACCOUNT_ID:root"},
            "Action": "sts:AssumeRole"
        }
    ]
}
```

```bash
# 다른 계정에서 역할 맡기
aws sts assume-role \
    --role-arn arn:aws:iam::TARGET_ACCOUNT:role/CrossAccountRole \
    --role-session-name MySession
```

---

## 3. GCP IAM

### 3.1 핵심 개념

```
┌─────────────────────────────────────────────────────────────┐
│  조직 (Organization)                                        │
│  ├── 폴더 (Folder)                                          │
│  │   └── 프로젝트 (Project)                                 │
│  │       └── 리소스 (Resource)                              │
│  └─────────────────────────────────────────────────────────│
│                                                             │
│  IAM 바인딩:                                                │
│  주체 (Member) + 역할 (Role) = 리소스에 대한 권한           │
│                                                             │
│  예: user:john@example.com + roles/storage.admin            │
│      → gs://my-bucket에 대한 관리자 권한                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 역할 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **기본 역할** | 넓은 권한 | Owner, Editor, Viewer |
| **사전정의 역할** | 서비스별 세분화 | roles/storage.admin |
| **커스텀 역할** | 사용자 정의 | my-custom-role |

### 3.3 역할 바인딩

```bash
# 프로젝트 수준 역할 부여
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin"

# 버킷 수준 역할 부여
gsutil iam ch user:john@example.com:objectViewer gs://my-bucket

# 역할 바인딩 조회
gcloud projects get-iam-policy PROJECT_ID

# 역할 제거
gcloud projects remove-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin"
```

### 3.4 서비스 계정

```bash
# 서비스 계정 생성
gcloud iam service-accounts create my-service-account \
    --display-name="My Service Account"

# 역할 부여
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:my-service-account@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# 키 파일 생성 (프로그래밍 접근)
gcloud iam service-accounts keys create key.json \
    --iam-account=my-service-account@PROJECT_ID.iam.gserviceaccount.com

# Compute Engine에 서비스 계정 연결
gcloud compute instances create my-instance \
    --service-account=my-service-account@PROJECT_ID.iam.gserviceaccount.com \
    --scopes=cloud-platform
```

### 3.5 워크로드 아이덴티티 (GKE)

```bash
# 워크로드 아이덴티티 풀 활성화
gcloud container clusters update my-cluster \
    --region=asia-northeast3 \
    --workload-pool=PROJECT_ID.svc.id.goog

# Kubernetes 서비스 계정과 GCP 서비스 계정 연결
gcloud iam service-accounts add-iam-policy-binding \
    my-gcp-sa@PROJECT_ID.iam.gserviceaccount.com \
    --role=roles/iam.workloadIdentityUser \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/K8S_SA]"
```

---

## 4. 최소 권한 원칙

### 4.1 원칙

```
최소 권한 = 작업 수행에 필요한 최소한의 권한만 부여

잘못된 예:
- Admin 권한을 모든 사용자에게
- * (모든 리소스)에 대한 권한

올바른 예:
- 필요한 Action만 명시
- 특정 리소스에 대한 권한
- 조건부 접근
```

### 4.2 AWS 정책 예시

**나쁜 예:**
```json
{
    "Effect": "Allow",
    "Action": "*",
    "Resource": "*"
}
```

**좋은 예:**
```json
{
    "Effect": "Allow",
    "Action": [
        "s3:GetObject",
        "s3:PutObject"
    ],
    "Resource": "arn:aws:s3:::my-bucket/uploads/*",
    "Condition": {
        "StringEquals": {
            "s3:x-amz-acl": "private"
        }
    }
}
```

### 4.3 GCP 역할 선택

```bash
# 권한이 너무 넓은 역할 (피할 것)
roles/owner
roles/editor

# 적절한 역할
roles/storage.objectViewer  # 객체 읽기만
roles/compute.instanceAdmin.v1  # 인스턴스 관리만
roles/cloudsql.client  # SQL 연결만
```

---

## 5. 조건부 접근

### 5.1 AWS 조건

```json
{
    "Effect": "Allow",
    "Action": "s3:*",
    "Resource": "*",
    "Condition": {
        "IpAddress": {
            "aws:SourceIp": "203.0.113.0/24"
        },
        "Bool": {
            "aws:MultiFactorAuthPresent": "true"
        },
        "DateGreaterThan": {
            "aws:CurrentTime": "2024-01-01T00:00:00Z"
        },
        "StringEquals": {
            "aws:RequestedRegion": "ap-northeast-2"
        }
    }
}
```

### 5.2 GCP 조건

```bash
# 조건부 역할 바인딩
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin" \
    --condition='expression=request.time < timestamp("2024-12-31T23:59:59Z"),title=Temporary Access,description=Access until end of year'

# IP 기반 조건 (VPC Service Controls와 함께)
expression: 'resource.name.startsWith("projects/PROJECT_ID/zones/asia-northeast3")'
```

---

## 6. 권한 분석

### 6.1 AWS IAM Access Analyzer

```bash
# Access Analyzer 생성
aws accessanalyzer create-analyzer \
    --analyzer-name my-analyzer \
    --type ACCOUNT

# 분석 결과 조회
aws accessanalyzer list-findings --analyzer-arn arn:aws:access-analyzer:...:analyzer/my-analyzer

# 정책 검증
aws accessanalyzer validate-policy \
    --policy-document file://policy.json \
    --policy-type IDENTITY_POLICY
```

### 6.2 GCP Policy Analyzer

```bash
# IAM 정책 분석
gcloud asset analyze-iam-policy \
    --organization=ORG_ID \
    --identity="user:john@example.com"

# 권한 확인
gcloud projects get-iam-policy PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:john@example.com" \
    --format="table(bindings.role)"
```

---

## 7. MFA (다중 인증)

### 7.1 AWS MFA

```bash
# 가상 MFA 활성화
aws iam create-virtual-mfa-device \
    --virtual-mfa-device-name john-mfa \
    --outfile qrcode.png \
    --bootstrap-method QRCodePNG

# MFA 디바이스 연결
aws iam enable-mfa-device \
    --user-name john \
    --serial-number arn:aws:iam::123456789012:mfa/john-mfa \
    --authentication-code1 123456 \
    --authentication-code2 789012

# MFA 필수 정책
{
    "Effect": "Deny",
    "Action": "*",
    "Resource": "*",
    "Condition": {
        "BoolIfExists": {
            "aws:MultiFactorAuthPresent": "false"
        }
    }
}
```

### 7.2 GCP 2단계 인증

```bash
# 조직 수준에서 2FA 강제 (Admin Console에서)
# Google Workspace Admin → Security → 2-Step Verification

# 서비스 계정은 MFA 불가 → 대신:
# - 키 파일 안전 관리
# - 워크로드 아이덴티티 사용
# - 단기 토큰 사용
```

---

## 8. 일반적인 역할 패턴

### 8.1 AWS 일반 역할

| 역할 | 권한 | 용도 |
|------|------|------|
| AdministratorAccess | 전체 | 관리자 |
| PowerUserAccess | IAM 제외 전체 | 개발자 |
| ReadOnlyAccess | 읽기 전용 | 감사/뷰어 |
| AmazonS3FullAccess | S3 전체 | 스토리지 관리 |
| AmazonEC2FullAccess | EC2 전체 | 컴퓨팅 관리 |

### 8.2 GCP 일반 역할

| 역할 | 권한 | 용도 |
|------|------|------|
| roles/owner | 전체 | 관리자 |
| roles/editor | IAM 제외 편집 | 개발자 |
| roles/viewer | 읽기 전용 | 뷰어 |
| roles/compute.admin | Compute 전체 | 인프라 관리 |
| roles/storage.admin | Storage 전체 | 스토리지 관리 |

---

## 9. 보안 모범 사례

```
□ Root/Owner 계정은 일상 업무에 사용하지 않음
□ Root/Owner 계정에 MFA 활성화
□ 최소 권한 원칙 적용
□ 그룹/역할을 통한 권한 관리 (개별 사용자 X)
□ 정기적인 권한 검토 (미사용 권한 제거)
□ 서비스 계정 키 파일 안전 관리
□ 임시 자격 증명 사용 (STS, 워크로드 아이덴티티)
□ 조건부 접근 활용 (IP, 시간, MFA)
□ 감사 로그 활성화 (CloudTrail, Cloud Audit Logs)
□ 정책 변경 알림 설정
```

---

## 10. 다음 단계

- [14_Security_Services.md](./14_Security_Services.md) - 보안 서비스
- [02_AWS_GCP_Account_Setup.md](./02_AWS_GCP_Account_Setup.md) - 계정 초기 설정

---

## 연습 문제

### 연습 문제 1: 최소 권한(Least Privilege) 정책 작성

Lambda 함수가 다음을 수행해야 합니다:
- S3 버킷 `my-app-data`에서 객체 읽기 (모든 객체)
- 특정 CloudWatch 로그 그룹 `/aws/lambda/my-function`에 로그 쓰기

최소 권한 원칙을 따르는 최소한의 IAM 정책 문서(JSON)를 작성하세요.

<details>
<summary>정답 보기</summary>

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ReadFromS3Bucket",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::my-app-data/*"
        },
        {
            "Sid": "WriteCloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:log-group:/aws/lambda/my-function:*"
        }
    ]
}
```

**이것이 최소 권한인 이유**:
- `s3:GetObject`만 — `s3:*`나 `s3:PutObject`가 아님. 함수는 읽기만 해야 하며, 쓰기나 삭제를 해서는 안 됩니다.
- S3 리소스는 `my-app-data/*`로 범위 제한 — `*`(모든 버킷)이 아님.
- CloudWatch 액션은 스트림 생성과 이벤트 저장으로 제한 — 로그 그룹 생성이나 리소스 조회 불가.
- 로그 그룹 리소스는 특정 함수의 로그 그룹으로 범위 제한.

**일반적인 실수**: CloudWatch 로그에 `"Resource": "*"` 사용 — 이것은 함수의 그룹만이 아닌 계정의 모든 로그 그룹에 대한 접근을 허용합니다.

</details>

### 연습 문제 2: EC2를 위한 IAM 역할(Role) vs IAM 사용자(User)

EC2 인스턴스가 S3 버킷에 파일을 업로드해야 합니다. 두 개발자가 다른 접근 방식을 제안합니다:
- **개발자 A**: S3 권한이 있는 IAM 사용자 생성, 액세스 키(access key) 생성, 애플리케이션 설정에 하드코딩
- **개발자 B**: S3 권한이 있는 IAM 역할 생성, EC2 인스턴스 프로파일에 연결

어떤 접근 방식이 올바릅니까? 잘못된 접근 방식의 보안 위험을 설명하세요.

<details>
<summary>정답 보기</summary>

**개발자 B가 올바릅니다** — 인스턴스 프로파일에 IAM 역할을 사용합니다.

**개발자 A의 접근 방식(IAM 사용자 + 액세스 키)의 문제점**:

1. **장기 자격 증명**: 액세스 키는 자동으로 만료되지 않습니다. 손상(코드, 설정 파일 또는 로그에서 누출)되면 수동으로 순환(rotate)할 때까지 무기한 사용될 수 있습니다.

2. **자격 증명 노출 위험**: 설정 파일에 자격 증명을 하드코딩하면 버전 관리(Git), Docker 이미지, 컨테이너 환경 변수, 또는 애플리케이션 로그에 포함될 수 있습니다.

3. **키 순환 오버헤드**: 액세스 키를 순환하려면 이를 사용하는 모든 서버와 설정을 업데이트해야 합니다 — 수동적이고 오류 발생 가능성이 있는 프로세스입니다.

**IAM 역할이 더 나은 이유**:
1. **임시 자격 증명**: EC2 메타데이터 서비스가 자동으로 순환되는 단기 자격 증명(1시간 STS 토큰)을 제공합니다. 도난되어도 빠르게 만료됩니다.
2. **관리할 시크릿(secret) 없음**: 애플리케이션이 인스턴스 메타데이터 서비스에서 자격 증명을 자동으로 가져오는 AWS SDK의 기본 자격 증명 체인을 사용합니다. 저장, 순환, 노출할 키가 없습니다.
3. **코드 변경 불필요**: 역할의 권한을 업데이트하여 더 제한적인 역할로 전환 — 애플리케이션 변경이나 배포가 필요 없습니다.

**AWS SDK 사용 방법**:
```python
import boto3

# 자격 증명 불필요 - SDK가 자동으로 인스턴스 역할 사용
s3 = boto3.client('s3')
s3.upload_file('local_file.txt', 'my-bucket', 'uploaded_file.txt')
```

</details>

### 연습 문제 3: 교차 계정(Cross-Account) 역할 가정(Role Assumption)

계정 A(ID: `111111111111`)에 중요한 데이터가 있는 S3 버킷이 있습니다. 계정 B(ID: `222222222222`)에 해당 버킷에서 읽어야 하는 Lambda 함수가 있습니다.

IAM 역할 가정(role assumption)을 사용한 교차 계정 접근에 필요한 완전한 설정을 설명하세요.

<details>
<summary>정답 보기</summary>

**1단계: 계정 A(리소스 계정)에 역할 생성**

계정 B의 Lambda가 가정할 수 있는 역할 생성:

```bash
# 계정 A에서
aws iam create-role \
    --role-name CrossAccountS3ReadRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::222222222222:root"
            },
            "Action": "sts:AssumeRole"
        }]
    }'

# 계정 A의 역할에 S3 읽기 정책 연결
aws iam attach-role-policy \
    --role-name CrossAccountS3ReadRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

**2단계: 계정 B의 Lambda에 역할 가정 권한 부여**

계정 B에서 Lambda 실행 역할에 정책 연결:
```json
{
    "Effect": "Allow",
    "Action": "sts:AssumeRole",
    "Resource": "arn:aws:iam::111111111111:role/CrossAccountS3ReadRole"
}
```

**3단계: Lambda 코드에서 역할 가정**

```python
import boto3

def lambda_handler(event, context):
    # 교차 계정 역할 가정
    sts = boto3.client('sts')
    assumed = sts.assume_role(
        RoleArn='arn:aws:iam::111111111111:role/CrossAccountS3ReadRole',
        RoleSessionName='LambdaCrossAccountSession'
    )

    # 임시 자격 증명을 사용하여 계정 A의 S3에 접근
    s3 = boto3.client('s3',
        aws_access_key_id=assumed['Credentials']['AccessKeyId'],
        aws_secret_access_key=assumed['Credentials']['SecretAccessKey'],
        aws_session_token=assumed['Credentials']['SessionToken']
    )

    response = s3.get_object(Bucket='account-a-bucket', Key='data.json')
    return response['Body'].read()
```

</details>

### 연습 문제 4: IAM 정책 분석

다음 IAM 정책을 분석하세요: 이 정책이 허용하고 거부하는 것은 무엇입니까? 보안 우려 사항이 있습니까?

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "*",
            "Resource": "*"
        },
        {
            "Effect": "Deny",
            "Action": [
                "iam:*",
                "organizations:*"
            ],
            "Resource": "*"
        }
    ]
}
```

<details>
<summary>정답 보기</summary>

**이 정책이 하는 일**:
- **Statement 1**: 모든 AWS 서비스와 리소스에 대한 완전한 접근(`*`) 허용 — `AdministratorAccess`와 동일
- **Statement 2**: 모든 IAM 액션과 모든 Organizations 액션을 명시적으로 거부

**실질적 효과**: 사용자는 IAM 관리(사용자/역할/정책 생성·삭제) 및 AWS Organizations 관리를 제외한 모든 것을 할 수 있습니다. IAM에서 Deny는 항상 Allow를 재정의합니다.

**보안 우려 사항**:

1. **여전히 위험하게 과잉 권한**: IAM 접근 없이도 사용자는 데이터베이스 삭제, EC2 인스턴스 종료, S3 버킷 비우기, Lambda 함수 배포, VPC 수정, 시크릿 접근 — 권한 변경을 제외한 거의 모든 것을 할 수 있습니다.

2. **IAM 거부는 불충분한 보호**: 사용자가 새 역할을 만들 수는 없지만, 기존 권한으로도 상당한 피해를 입히거나 데이터를 유출할 수 있습니다.

3. **"IAM을 거부하여 권한 상승 방지" 패턴은 취약**: 넓은 권한을 가진 사용자는 여전히 다른 방법(예: Lambda, EC2 User Data, CloudFormation을 통해 높은 권한으로 코드 실행)으로 권한을 상승시킬 수 있습니다.

**더 나은 접근 방식**: 권한이 없는 상태에서 시작하여 필요한 것만 추가(허용 목록)하는 것이 맞으며, 모든 권한으로 시작하여 특정 액션을 거부(거부 목록)하는 것은 최소 권한 원칙을 위반합니다.

</details>

### 연습 문제 5: GCP 서비스 계정(Service Account) 모범 사례

GCP Compute Engine 인스턴스가 Cloud Storage에 지표를 쓰고 Secret Manager에서 설정을 읽는 Python 애플리케이션을 실행합니다. 최소한의 서비스 계정 설정을 생성하고 올바른 사전 정의된 역할을 바인딩하세요.

<details>
<summary>정답 보기</summary>

```bash
# 1단계: 전용 서비스 계정 생성 (애플리케이션당 하나)
gcloud iam service-accounts create my-app-sa \
    --display-name="My Application Service Account" \
    --project=my-project-id

# 2단계: 최소 필요 역할 부여

# Storage Object Creator: Cloud Storage에 객체 쓰기 허용 (읽기, 버킷 삭제 불가)
gcloud projects add-iam-policy-binding my-project-id \
    --member="serviceAccount:my-app-sa@my-project-id.iam.gserviceaccount.com" \
    --role="roles/storage.objectCreator" \
    --condition="expression=resource.name.startsWith('projects/_/buckets/my-metrics-bucket'),title=only-metrics-bucket"

# Secret Accessor: 시크릿 값 읽기 허용 (시크릿 생성 또는 관리 불가)
gcloud projects add-iam-policy-binding my-project-id \
    --member="serviceAccount:my-app-sa@my-project-id.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# 3단계: Compute Engine 인스턴스에 서비스 계정 할당
gcloud compute instances create my-instance \
    --service-account=my-app-sa@my-project-id.iam.gserviceaccount.com \
    --scopes=cloud-platform \
    --zone=asia-northeast3-a
```

**모범 사례 참고사항**:
- **애플리케이션당 하나의 서비스 계정** — 애플리케이션 간에 서비스 계정을 공유하지 마세요. 하나가 손상되면 해당 애플리케이션의 권한만 노출됩니다.
- 가능한 경우 **IAM 조건**을 사용하여 특정 리소스(예: 특정 버킷만)로 범위를 제한합니다.
- `roles/storage.admin` 대신 **`roles/storage.objectCreator`** — 새 객체만 쓰기, 삭제나 다른 객체 읽기 불가.
- `roles/secretmanager.admin` 대신 **`roles/secretmanager.secretAccessor`** — 시크릿 값만 읽기, 시크릿 생성 또는 수정 불가.
- **서비스 계정 키 파일 다운로드 방지** — Compute Engine 인스턴스에는 대신 메타데이터 서버를 사용하세요. 키 파일은 수동으로 순환해야 하는 장기 자격 증명입니다.

</details>

---

## 참고 자료

- [AWS IAM Documentation](https://docs.aws.amazon.com/iam/)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [GCP IAM Documentation](https://cloud.google.com/iam/docs)
- [GCP IAM Best Practices](https://cloud.google.com/iam/docs/using-iam-securely)
