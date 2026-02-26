# AWS & GCP 계정 설정

**이전**: [클라우드 컴퓨팅 개요](./01_Cloud_Computing_Overview.md) | **다음**: [리전과 가용 영역](./03_Regions_Availability_Zones.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 적절한 보안 설정을 갖춘 AWS 계정을 생성하고 구성할 수 있습니다
2. 결제가 활성화된 GCP 프로젝트를 생성하고 구성할 수 있습니다
3. Root 및 관리자 계정에 다중 인증(MFA, Multi-Factor Authentication)을 활성화할 수 있습니다
4. 두 플랫폼 모두에서 결제 알림 및 예산 임계값을 설정할 수 있습니다
5. 일상적인 Root 계정 사용을 피하기 위해 IAM 관리자 사용자를 구성할 수 있습니다
6. AWS Management Console과 GCP Cloud Console을 탐색할 수 있습니다

---

클라우드 리소스를 배포하기 전에 올바르게 보안이 설정된 계정이 필요합니다. 잘못된 계정 설정은 보안 침해와 예상치 못한 청구서의 가장 흔한 원인 중 하나입니다. 이 레슨은 AWS와 GCP 계정을 생성하고, 보안을 강화하고, 구성하는 정확한 단계를 안내하여 첫날부터 탄탄한 기반을 구축할 수 있도록 합니다.

## 1. AWS 계정 생성

### 1.1 계정 생성 절차

1. **AWS 가입 페이지 접속**
   - https://aws.amazon.com/ 에서 "Create an AWS Account" 클릭

2. **계정 정보 입력**
   - 이메일 주소 (Root 계정용)
   - AWS 계정 이름
   - 비밀번호

3. **연락처 정보**
   - 계정 유형: 개인(Personal) 또는 비즈니스(Business)
   - 이름, 주소, 전화번호

4. **결제 정보**
   - 신용카드 등록 (무료 티어 사용 시에도 필수)
   - $1 인증 결제 후 환불됨

5. **본인 확인**
   - SMS 또는 음성 통화로 PIN 확인

6. **지원 플랜 선택**
   - Basic Support (무료) 선택 권장

### 1.2 Root 계정 보안

Root 계정은 모든 권한을 가지므로 반드시 보안 강화가 필요합니다.

```
⚠️ Root 계정 보안 체크리스트
□ 강력한 비밀번호 설정 (16자 이상, 특수문자 포함)
□ MFA(다중 인증) 활성화
□ 액세스 키 생성 금지
□ 일상 업무에 Root 계정 사용 금지
□ IAM 사용자 생성하여 사용
```

### 1.3 MFA 설정 (AWS)

**Console에서 MFA 활성화:**

1. AWS Console → 우측 상단 계정명 → "Security credentials"
2. "Multi-factor authentication (MFA)" 섹션
3. "Activate MFA" 클릭
4. MFA 디바이스 유형 선택:
   - **Virtual MFA device**: Google Authenticator, Authy 앱 사용
   - **Hardware TOTP token**: 물리적 토큰
   - **Security key**: FIDO 보안 키

**Virtual MFA 설정:**
```
1. 앱 설치: Google Authenticator 또는 Authy
2. 앱에서 QR 코드 스캔
3. 연속된 두 개의 MFA 코드 입력
4. "Assign MFA" 클릭
```

---

## 2. GCP 계정 생성

### 2.1 계정 생성 절차

1. **GCP Console 접속**
   - https://console.cloud.google.com/

2. **Google 계정 로그인**
   - 기존 Google 계정 사용 또는 새로 생성

3. **GCP 이용약관 동의**
   - 국가 선택
   - 서비스 약관 동의

4. **결제 계정 설정** (무료 체험용)
   - 신용카드 정보 입력
   - $300 무료 크레딧 활성화 (90일)

5. **첫 번째 프로젝트 생성**
   - 프로젝트명 지정
   - 조직 선택 (개인은 "조직 없음")

### 2.2 GCP 보안 설정

**Google 계정 보안 강화:**

```
⚠️ GCP 계정 보안 체크리스트
□ Google 계정에 2단계 인증 활성화
□ 비밀번호 보안 강화
□ 복구 이메일/전화번호 설정
□ 조직 정책 검토 (비즈니스)
□ 서비스 계정 사용 권장
```

### 2.3 2단계 인증 설정 (GCP)

1. Google 계정 설정 → "보안"
2. "2단계 인증" 활성화
3. 인증 방법 선택:
   - Google 메시지
   - 인증 앱 (Google Authenticator)
   - 보안 키
   - 백업 코드

---

## 3. 콘솔 탐색

### 3.1 AWS Management Console

**주요 UI 구성:**

```
┌─────────────────────────────────────────────────────────────┐
│  [AWS 로고]  서비스 검색창              리전 ▼  계정 ▼     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [서비스 메뉴]                                              │
│   ├── Compute (EC2, Lambda, ECS...)                         │
│   ├── Storage (S3, EBS, EFS...)                             │
│   ├── Database (RDS, DynamoDB...)                           │
│   ├── Networking (VPC, Route 53...)                         │
│   ├── Security (IAM, KMS...)                                │
│   └── Management (CloudWatch, CloudFormation...)            │
│                                                             │
│  [최근 방문 서비스]                                         │
│  [즐겨찾기 서비스]                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**유용한 기능:**
- **서비스 검색**: 상단 검색창에 서비스명 입력
- **리전 선택**: 우측 상단에서 작업할 리전 선택
- **CloudShell**: 브라우저 내 터미널 (AWS CLI 사전 설치)
- **Resource Groups**: 리소스 그룹화 및 태그 관리

### 3.2 GCP Console

**주요 UI 구성:**

```
┌─────────────────────────────────────────────────────────────┐
│  [GCP 로고]  [프로젝트 선택 ▼]  검색창         [계정 아이콘]│
├──────────────┬──────────────────────────────────────────────┤
│  [네비게이션]│                                              │
│   │          │  [대시보드]                                   │
│   ├─ 컴퓨팅   │   ├── 프로젝트 정보                          │
│   ├─ 스토리지 │   ├── 리소스 요약                            │
│   ├─ 네트워킹 │   ├── API 활동                               │
│   ├─ 데이터베이스│ └── 빌링 요약                             │
│   ├─ 보안     │                                              │
│   ├─ 도구     │                                              │
│   └─ 빌링     │                                              │
│              │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

**유용한 기능:**
- **프로젝트 선택**: 좌측 상단에서 프로젝트 전환
- **Cloud Shell**: 우측 상단 터미널 아이콘 (gcloud 사전 설치)
- **핀 고정**: 자주 사용하는 서비스를 메뉴에 고정
- **API 및 서비스**: API 활성화 관리

---

## 4. 첫 번째 프로젝트/리소스 그룹

### 4.1 AWS: 태그를 통한 리소스 관리

AWS는 프로젝트 개념 대신 **태그**로 리소스를 관리합니다.

```bash
# 리소스 생성 시 태그 지정 예시
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type t2.micro \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=MyApp},{Key=Environment,Value=dev}]'
```

**태그 모범 사례:**

| 태그 키 | 예시 값 | 용도 |
|--------|--------|------|
| Project | MyApp | 프로젝트별 비용 추적 |
| Environment | dev, staging, prod | 환경 구분 |
| Owner | john@example.com | 담당자 식별 |
| CostCenter | IT-001 | 비용 센터 할당 |

### 4.2 GCP: 프로젝트 생성

GCP는 **프로젝트** 단위로 리소스와 빌링을 격리합니다.

**프로젝트 생성:**
1. Console 상단 → 프로젝트 선택 드롭다운
2. "새 프로젝트" 클릭
3. 프로젝트 이름 입력 (고유한 ID 자동 생성)
4. 빌링 계정 연결
5. "만들기" 클릭

```bash
# gcloud로 프로젝트 생성
gcloud projects create my-project-id \
    --name="My Project" \
    --labels=env=dev

# 프로젝트 전환
gcloud config set project my-project-id
```

**프로젝트 구조 권장:**

```
Organization (선택)
├── Folder: Development
│   ├── Project: dev-frontend
│   └── Project: dev-backend
├── Folder: Production
│   ├── Project: prod-frontend
│   └── Project: prod-backend
└── Folder: Shared
    └── Project: shared-services
```

---

## 5. 비용 알림 설정

### 5.1 AWS 예산 알림

**AWS Budgets 설정:**

1. AWS Console → "Billing and Cost Management" → "Budgets"
2. "Create budget" 클릭
3. 예산 유형 선택: "Cost budget"
4. 예산 설정:
   - 이름: "Monthly Budget"
   - 금액: 원하는 한도 (예: $50)
   - 기간: Monthly

5. 알림 조건:
   - 실제 비용이 예산의 80% 도달 시 알림
   - 예측 비용이 100% 초과 시 알림

6. 알림 수신:
   - 이메일 주소 입력
   - SNS 토픽 연결 (선택)

```bash
# AWS CLI로 예산 생성
aws budgets create-budget \
    --account-id 123456789012 \
    --budget '{
        "BudgetName": "Monthly-50USD",
        "BudgetLimit": {"Amount": "50", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[{
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80
        },
        "Subscribers": [{
            "SubscriptionType": "EMAIL",
            "Address": "your@email.com"
        }]
    }]'
```

### 5.2 GCP 예산 알림

**GCP Billing 예산 설정:**

1. Console → "결제" → "예산 및 알림"
2. "예산 만들기" 클릭
3. 예산 설정:
   - 이름: "Monthly Budget"
   - 프로젝트: 전체 또는 특정 프로젝트
   - 금액: 지정 금액 (예: $50)

4. 알림 임계값:
   - 50%, 90%, 100% 알림 설정

5. 알림 채널:
   - 이메일 수신자
   - Cloud Monitoring (선택)
   - Pub/Sub 토픽 (자동화용)

```bash
# gcloud로 예산 생성
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Monthly Budget" \
    --budget-amount=50USD \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0
```

---

## 6. 무료 티어 활용

### 6.1 AWS 무료 티어

| 유형 | 서비스 | 무료 한도 |
|------|--------|----------|
| **12개월 무료** | EC2 | t2.micro 750시간/월 |
| | S3 | 5GB 스토리지 |
| | RDS | db.t2.micro 750시간/월 |
| | CloudFront | 50GB 데이터 전송 |
| **항상 무료** | Lambda | 100만 요청/월 |
| | DynamoDB | 25GB 스토리지, 25 WCU/RCU |
| | SNS | 100만 요청/월 |
| | CloudWatch | 기본 모니터링 |

**무료 티어 모니터링:**
- Console → "Billing" → "Free Tier" 탭에서 사용량 확인

### 6.2 GCP 무료 티어

| 유형 | 서비스 | 무료 한도 |
|------|--------|----------|
| **$300 크레딧** | 모든 서비스 | 90일간 (신규 계정) |
| **Always Free** | Compute Engine | e2-micro 1개 (특정 리전) |
| | Cloud Storage | 5GB (US 리전) |
| | Cloud Functions | 200만 호출/월 |
| | BigQuery | 1TB 쿼리/월, 10GB 스토리지 |
| | Cloud Run | 200만 요청/월 |
| | Firestore | 1GB 스토리지, 50K 읽기/일 |

**Always Free 리전 제한:**
- Compute Engine e2-micro: us-west1, us-central1, us-east1만 해당

---

## 7. 초기 보안 설정 요약

### 7.1 AWS 초기 보안 체크리스트

```
□ Root 계정 MFA 활성화
□ Root 액세스 키 삭제 확인
□ IAM 사용자 생성 및 MFA 활성화
□ IAM 비밀번호 정책 강화
□ CloudTrail 활성화 (감사 로그)
□ 예산 알림 설정
□ S3 퍼블릭 액세스 차단 설정 확인
```

### 7.2 GCP 초기 보안 체크리스트

```
□ Google 계정 2단계 인증 활성화
□ 조직 정책 검토 (해당 시)
□ 서비스 계정 생성 (애플리케이션용)
□ 최소 권한 IAM 역할 부여
□ Cloud Audit Logs 활성화
□ 예산 알림 설정
□ VPC 방화벽 규칙 검토
```

---

## 8. 다음 단계

- [03_Regions_Availability_Zones.md](./03_Regions_Availability_Zones.md) - 리전과 가용 영역 이해
- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM 상세 설정

---

## 연습 문제

### 연습 문제 1: 루트 계정(Root Account) 보안 감사

방금 새 AWS 계정을 생성했고 사용 전에 보안을 강화하려 합니다. 즉시 수행해야 할 모든 단계를 올바른 우선순위 순서로 나열하고 각 단계가 중요한 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

1. **루트 계정에 MFA(다중 인증, Multi-Factor Authentication) 활성화** — 루트 사용자는 모든 것에 대한 무제한 접근 권한을 가집니다. MFA 없이 침해되면 공격자가 모든 리소스를 삭제하고 데이터를 빼가며 막대한 요금을 발생시킬 수 있습니다.
2. **루트 액세스 키(access key) 삭제 또는 생성 금지** — 액세스 키는 프로그래밍 방식 접근을 허용합니다. 루트 액세스 키는 권한 정책을 우회하므로 특히 위험합니다.
3. **IAM 관리자 사용자 생성** — 일상적인 작업에 루트 계정을 절대 사용하지 마세요. `AdministratorAccess` 정책이 부여된 IAM 사용자를 생성하여 사용합니다.
4. **IAM 관리자 사용자에 MFA 활성화** — 관리자 IAM 사용자도 고가치 타깃입니다. MFA가 중요한 두 번째 인증 요소를 추가합니다.
5. **청구(billing) 알림 설정** — 실수로 인한 리소스 생성이나 계정 침해로 인한 예상치 못한 청구를 방지합니다.
6. **CloudTrail 활성화** — 모든 API 활동에 대한 감사 로그(audit log)를 생성하며, 무단 접근 탐지에 필수적입니다.

</details>

### 연습 문제 2: GCP vs AWS 계정 구조

팀이 개발(development), 스테이징(staging), 운영(production) 세 가지 환경을 위한 클라우드 인프라를 설정하고 있습니다.

1. GCP에서 이 환경들을 어떻게 구성하겠습니까? 권장 프로젝트 구조를 설명하세요.
2. AWS에서 동일한 격리를 어떻게 처리하겠습니까? GCP 프로젝트를 대체하는 메커니즘은 무엇입니까?

<details>
<summary>정답 보기</summary>

1. **GCP 구조** — 폴더(folder)와 프로젝트(project) 계층 구조를 사용합니다:
   ```
   Organization
   ├── Folder: Development
   │   └── Project: myapp-dev
   ├── Folder: Staging
   │   └── Project: myapp-staging
   └── Folder: Production
       └── Project: myapp-prod
   ```
   각 프로젝트에는 자체 청구, IAM 정책, 리소스 할당량이 있어 강력한 격리를 제공합니다.

2. **AWS 구조** — AWS는 격리를 위해 **태그(tag)** 와 선택적으로 **AWS Organizations**(별도 계정)를 사용합니다:
   - 단일 계정 내 경량 분리: 모든 리소스에 일관된 태그(`Environment=dev/staging/prod`)를 적용하고 IAM 정책과 리소스 그룹으로 경계를 강제합니다.
   - 강력한 격리: **AWS Organizations**를 사용하여 환경별로 별도 AWS 계정을 사용합니다(프로덕션 권장). 각 계정에는 자체 IAM, 청구, 리소스 한도가 있습니다.
   - 핵심 차이점: GCP의 프로젝트는 내장된 일급(first-class) 개념이고, AWS는 동등한 격리를 달성하기 위해 의도적인 태그 규율 또는 별도 계정이 필요합니다.

</details>

### 연습 문제 3: 예산 알림(Budget Alert) 설정

개발자가 클라우드를 학습 중이며 한 달에 20달러 이상 실수로 지출하지 않으려 합니다. AWS CLI 명령어를 작성하여 예산 한도의 80%에서 이메일 알림을 보내는 예산을 생성하세요. 수신자로 `your@email.com`을 사용하세요.

<details>
<summary>정답 보기</summary>

```bash
aws budgets create-budget \
    --account-id YOUR_ACCOUNT_ID \
    --budget '{
        "BudgetName": "Monthly-20USD",
        "BudgetLimit": {"Amount": "20", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[{
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80
        },
        "Subscribers": [{
            "SubscriptionType": "EMAIL",
            "Address": "your@email.com"
        }]
    }]'
```

**핵심 파라미터 설명**:
- `"Amount": "20"` — $20 한도 설정
- `"TimeUnit": "MONTHLY"` — 매월 초기화
- `"NotificationType": "ACTUAL"` — 실제(예측이 아닌) 지출에 알림
- `"Threshold": 80` — 80%($16)에서 발동

100%(예측) 알림도 추가하려면 `notifications-with-subscribers` 배열에 `"NotificationType": "FORECASTED"` 및 `"Threshold": 100`으로 두 번째 객체를 추가합니다.

</details>

### 연습 문제 4: 무료 티어(Free Tier) 계획

개인 프로젝트를 구축 중이며 첫 1년 동안 무료 티어(free tier) 내에서만 사용하려 합니다. 애플리케이션에는 웹 서버, 관계형 데이터베이스(relational database), 사용자 업로드를 위한 파일 스토리지가 필요합니다.

AWS에서 무료 티어 내에 머물기 위해 선택할 특정 서비스와 인스턴스 유형/설정은 무엇입니까? 주의해야 할 제한 사항을 나열하세요.

<details>
<summary>정답 보기</summary>

| 필요 | AWS 서비스 | 무료 티어 한도 | 주요 제한 사항 |
|------|------------|----------------|----------------|
| 웹 서버 | EC2 **t2.micro** | 월 750시간 (12개월) | RAM 1GB; 750시간을 초과하지 않으려면 여러 인스턴스 실행 시 주의 |
| 관계형 데이터베이스 | RDS **db.t2.micro** | 월 750시간 (12개월) | 20GB 스토리지; Single-AZ(단일 가용 영역)만 가능; MySQL, PostgreSQL, MariaDB, SQL Server Express 지원 |
| 파일 스토리지 | S3 | 5GB 표준 스토리지, 월 20,000 GET 요청, 2,000 PUT 요청 (12개월) | 트래픽이 많은 앱의 요청 횟수 주의; 아웃바운드(egress) 데이터 전송은 월 100GB 초과 시 무료 아님 |

**중요 주의사항**:
- 무료 티어 한도는 서비스 인스턴스별이 아닌 계정별로 적용됩니다. t2.micro 인스턴스 두 개를 실행하면 1,500시간을 소비하여 750시간 한도를 초과합니다.
- 12개월 무료 티어는 서비스 최초 사용 시점이 아닌 계정 생성 시점부터 시작됩니다.
- 항상 AWS 청구에서 무료 티어 사용 알림(Free Tier Usage Alert)을 활성화하여 요금 발생 전에 알림을 받으세요.

</details>

### 연습 문제 5: MFA 방식 비교

AWS에서 사용 가능한 세 가지 MFA 디바이스 유형(가상 MFA, 하드웨어 TOTP 토큰, 보안 키)을 비교하세요. 각각이 가장 적합한 사용 사례를 설명하세요.

<details>
<summary>정답 보기</summary>

| MFA 유형 | 동작 방식 | 가장 적합한 사용 사례 |
|----------|----------|----------------------|
| **가상 MFA 디바이스** (예: Google Authenticator, Authy) | 스마트폰 앱에서 생성되는 시간 기반 일회용 비밀번호(TOTP, Time-based One-Time Password) | 개인 개발자와 개인 계정. 무료이고 편리하며 추가 하드웨어가 필요 없음. 학습 및 개발 환경에 최적 |
| **하드웨어 TOTP 토큰** (예: Gemalto 토큰) | TOTP 코드를 생성하는 전용 물리 디바이스 | 직원이 업무용 MFA에 개인 휴대폰을 사용하지 않아야 하는 기업 환경, 또는 보안상 별도의 비네트워크 디바이스가 필요한 상황 |
| **보안 키** (FIDO2/WebAuthn, 예: YubiKey) | 암호화 인증을 수행하는 물리적 USB/NFC 키 | 피싱(phishing) 저항성이 중요한 고보안 계정(루트 계정, 비상 관리자 계정). 보안 키는 도메인을 검증하므로 피싱에 면역. 프로덕션 루트 계정에 최적 |

**대부분의 팀에 대한 권장사항**: 개발 중 IAM 사용자에게는 가상 MFA 디바이스를 사용하고, 루트 계정과 특권 관리자 계정에는 하드웨어 보안 키(YubiKey)를 사용하세요.

</details>

---

## 참고 자료

- [AWS 계정 생성 가이드](https://docs.aws.amazon.com/accounts/latest/reference/manage-acct-creating.html)
- [GCP 시작하기](https://cloud.google.com/docs/get-started)
- [AWS 무료 티어](https://aws.amazon.com/free/)
- [GCP 무료 티어](https://cloud.google.com/free)
