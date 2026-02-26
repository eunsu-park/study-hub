# 보안 서비스

**이전**: [IAM](./13_Identity_Access_Management.md) | **다음**: [CLI & SDK](./15_CLI_and_SDK.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 인프라부터 애플리케이션 수준까지 계층형 보안 모델(layered security model)을 설명할 수 있습니다
2. 네트워크, 데이터, 애플리케이션 계층에 걸쳐 AWS와 GCP 보안 서비스를 비교할 수 있습니다
3. 관리형 키 서비스(KMS)를 사용하여 저장 시 암호화(encryption at rest)와 전송 중 암호화(encryption in transit)를 구성할 수 있습니다
4. 웹 애플리케이션 방화벽(WAF)과 DDoS 보호(Shield/Cloud Armor)를 구현할 수 있습니다
5. 취약점 스캐닝(Inspector, Security Command Center) 및 컴플라이언스 감사 도구를 사용할 수 있습니다
6. 여러 클라우드 보안 서비스를 활용하여 심층 방어(defense-in-depth) 보안 아키텍처를 설계할 수 있습니다

---

클라우드 제공업체는 기본 방화벽을 훨씬 뛰어넘는 풍부한 보안 서비스를 제공합니다. 암호화 키 관리와 DDoS 보호부터 자동화된 취약점 스캐닝 및 컴플라이언스 모니터링에 이르기까지, 이러한 도구들은 심층 방어(defense-in-depth) 전략의 계층을 형성합니다. 어떤 서비스가 존재하고 어떻게 서로 맞물리는지 이해하는 것은 진화하는 위협으로부터 클라우드 워크로드를 보호하는 데 필수적입니다.

## 1. 보안 개요

### 1.1 클라우드 보안 계층

```
┌─────────────────────────────────────────────────────────────┐
│  애플리케이션 보안                                          │
│  - 입력 검증, 인증/인가, 세션 관리                          │
├─────────────────────────────────────────────────────────────┤
│  데이터 보안                                                │
│  - 암호화 (저장 시, 전송 중), 키 관리                       │
├─────────────────────────────────────────────────────────────┤
│  네트워크 보안                                              │
│  - 방화벽, VPC, WAF, DDoS 보호                              │
├─────────────────────────────────────────────────────────────┤
│  인프라 보안                                                │
│  - 패치 관리, 취약점 스캐닝                                 │
├─────────────────────────────────────────────────────────────┤
│  ID/접근 관리                                               │
│  - IAM, MFA, 최소 권한                                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 서비스 매핑

| 기능 | AWS | GCP |
|------|-----|-----|
| 네트워크 방화벽 | Security Groups, NACL | Firewall Rules |
| WAF | AWS WAF | Cloud Armor |
| DDoS | AWS Shield | Cloud Armor |
| 키 관리 | KMS | Cloud KMS |
| 비밀 관리 | Secrets Manager | Secret Manager |
| 취약점 스캐닝 | Inspector | Security Command Center |
| 위협 탐지 | GuardDuty | Security Command Center |

---

## 2. 네트워크 보안

### 2.1 AWS Security Groups

```bash
# 보안 그룹 생성
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server SG" \
    --vpc-id vpc-12345678

# 인바운드 규칙 추가
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --ip-permissions '[
        {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "203.0.113.0/24", "Description": "Office IP"}]}
    ]'

# 다른 보안 그룹에서 오는 트래픽 허용
aws ec2 authorize-security-group-ingress \
    --group-id sg-db \
    --source-group sg-app \
    --protocol tcp \
    --port 3306

# 규칙 삭제
aws ec2 revoke-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0
```

### 2.2 GCP Firewall Rules

```bash
# 방화벽 규칙 생성
gcloud compute firewall-rules create allow-http \
    --network=my-vpc \
    --allow=tcp:80,tcp:443 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=http-server \
    --priority=1000

# SSH 허용 (특정 IP)
gcloud compute firewall-rules create allow-ssh-office \
    --network=my-vpc \
    --allow=tcp:22 \
    --source-ranges=203.0.113.0/24 \
    --target-tags=ssh-server

# 내부 통신 허용
gcloud compute firewall-rules create allow-internal \
    --network=my-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/8

# 거부 규칙 (낮은 우선순위)
gcloud compute firewall-rules create deny-all-ingress \
    --network=my-vpc \
    --action=DENY \
    --rules=all \
    --source-ranges=0.0.0.0/0 \
    --priority=65534

# 규칙 삭제
gcloud compute firewall-rules delete allow-http
```

---

## 3. WAF (Web Application Firewall)

### 3.1 AWS WAF

```bash
# 웹 ACL 생성
aws wafv2 create-web-acl \
    --name my-web-acl \
    --scope REGIONAL \
    --default-action Allow={} \
    --visibility-config SampledRequestsEnabled=true,CloudWatchMetricsEnabled=true,MetricName=my-web-acl \
    --rules '[
        {
            "Name": "AWSManagedRulesCommonRuleSet",
            "Priority": 1,
            "OverrideAction": {"None": {}},
            "Statement": {
                "ManagedRuleGroupStatement": {
                    "VendorName": "AWS",
                    "Name": "AWSManagedRulesCommonRuleSet"
                }
            },
            "VisibilityConfig": {
                "SampledRequestsEnabled": true,
                "CloudWatchMetricsEnabled": true,
                "MetricName": "CommonRules"
            }
        }
    ]'

# ALB에 연결
aws wafv2 associate-web-acl \
    --web-acl-arn arn:aws:wafv2:...:webacl/my-web-acl/xxx \
    --resource-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx
```

**일반 규칙:**
- AWSManagedRulesCommonRuleSet: OWASP Top 10
- AWSManagedRulesSQLiRuleSet: SQL 인젝션
- AWSManagedRulesKnownBadInputsRuleSet: 악성 입력
- AWSManagedRulesAmazonIpReputationList: IP 평판

### 3.2 GCP Cloud Armor

```bash
# 보안 정책 생성
gcloud compute security-policies create my-policy \
    --description="My security policy"

# 규칙 추가 (SQL 인젝션 차단)
gcloud compute security-policies rules create 1000 \
    --security-policy=my-policy \
    --expression="evaluatePreconfiguredWaf('sqli-v33-stable')" \
    --action=deny-403

# 규칙 추가 (XSS 차단)
gcloud compute security-policies rules create 2000 \
    --security-policy=my-policy \
    --expression="evaluatePreconfiguredWaf('xss-v33-stable')" \
    --action=deny-403

# 규칙 추가 (IP 차단)
gcloud compute security-policies rules create 3000 \
    --security-policy=my-policy \
    --src-ip-ranges="203.0.113.0/24" \
    --action=deny-403

# 속도 제한
gcloud compute security-policies rules create 4000 \
    --security-policy=my-policy \
    --expression="true" \
    --action=rate-based-ban \
    --rate-limit-threshold-count=1000 \
    --rate-limit-threshold-interval-sec=60

# 백엔드 서비스에 연결
gcloud compute backend-services update my-backend \
    --security-policy=my-policy \
    --global
```

---

## 4. 키 관리 (KMS)

### 4.1 AWS KMS

```bash
# 고객 관리 키 생성
aws kms create-key \
    --description "My encryption key" \
    --key-usage ENCRYPT_DECRYPT \
    --origin AWS_KMS

# 별칭 생성
aws kms create-alias \
    --alias-name alias/my-key \
    --target-key-id 12345678-1234-1234-1234-123456789012

# 데이터 암호화
aws kms encrypt \
    --key-id alias/my-key \
    --plaintext fileb://plaintext.txt \
    --output text \
    --query CiphertextBlob | base64 --decode > encrypted.bin

# 데이터 복호화
aws kms decrypt \
    --ciphertext-blob fileb://encrypted.bin \
    --output text \
    --query Plaintext | base64 --decode > decrypted.txt

# 키 정책 업데이트
aws kms put-key-policy \
    --key-id 12345678-1234-1234-1234-123456789012 \
    --policy-name default \
    --policy file://key-policy.json
```

**S3 서버 측 암호화:**
```bash
# 버킷 암호화 설정
aws s3api put-bucket-encryption \
    --bucket my-bucket \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "aws:kms",
                "KMSMasterKeyID": "alias/my-key"
            }
        }]
    }'
```

### 4.2 GCP Cloud KMS

```bash
# 키 링 생성
gcloud kms keyrings create my-keyring \
    --location=asia-northeast3

# 암호화 키 생성
gcloud kms keys create my-key \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --purpose=encryption

# 데이터 암호화
gcloud kms encrypt \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --key=my-key \
    --plaintext-file=plaintext.txt \
    --ciphertext-file=encrypted.bin

# 데이터 복호화
gcloud kms decrypt \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --key=my-key \
    --ciphertext-file=encrypted.bin \
    --plaintext-file=decrypted.txt

# 서비스 계정에 암호화 권한 부여
gcloud kms keys add-iam-policy-binding my-key \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --member="serviceAccount:my-sa@PROJECT.iam.gserviceaccount.com" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

**Cloud Storage CMEK:**
```bash
# 버킷 생성 시 CMEK 지정
gsutil mb -l asia-northeast3 \
    -k projects/PROJECT/locations/asia-northeast3/keyRings/my-keyring/cryptoKeys/my-key \
    gs://my-encrypted-bucket
```

---

## 5. 비밀 관리

### 5.1 AWS Secrets Manager

```bash
# 비밀 생성
aws secretsmanager create-secret \
    --name my-database-credentials \
    --secret-string '{"username":"admin","password":"MySecretPassword123!"}'

# 비밀 조회
aws secretsmanager get-secret-value \
    --secret-id my-database-credentials \
    --query SecretString \
    --output text

# 비밀 업데이트
aws secretsmanager update-secret \
    --secret-id my-database-credentials \
    --secret-string '{"username":"admin","password":"NewPassword456!"}'

# 자동 로테이션 설정
aws secretsmanager rotate-secret \
    --secret-id my-database-credentials \
    --rotation-lambda-arn arn:aws:lambda:...:function:RotateSecret \
    --rotation-rules AutomaticallyAfterDays=30
```

**애플리케이션에서 사용:**
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

credentials = get_secret('my-database-credentials')
db_password = credentials['password']
```

### 5.2 GCP Secret Manager

```bash
# 비밀 생성
echo -n "MySecretPassword123!" | gcloud secrets create my-secret \
    --data-file=-

# 또는 파일에서
gcloud secrets create my-secret --data-file=secret.txt

# 비밀 조회
gcloud secrets versions access latest --secret=my-secret

# 새 버전 추가
echo -n "NewPassword456!" | gcloud secrets versions add my-secret \
    --data-file=-

# 서비스 계정에 접근 권한 부여
gcloud secrets add-iam-policy-binding my-secret \
    --member="serviceAccount:my-sa@PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**애플리케이션에서 사용:**
```python
from google.cloud import secretmanager

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/PROJECT_ID/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

password = get_secret('my-secret')
```

---

## 6. 암호화

### 6.1 저장 시 암호화 (Encryption at Rest)

| 서비스 | AWS | GCP |
|--------|-----|-----|
| 객체 스토리지 | S3 SSE-S3, SSE-KMS | Cloud Storage CMEK |
| 블록 스토리지 | EBS 암호화 | PD 암호화 |
| 데이터베이스 | RDS 암호화 | Cloud SQL 암호화 |
| 기본 암호화 | 일부 서비스 기본 | 모든 서비스 기본 |

### 6.2 전송 중 암호화 (Encryption in Transit)

```bash
# AWS ALB HTTPS 강제
aws elbv2 modify-listener \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --protocol HTTPS \
    --certificates CertificateArn=arn:aws:acm:...:certificate/xxx

# GCP HTTPS 로드밸런서
gcloud compute target-https-proxies create my-https-proxy \
    --url-map=my-url-map \
    --ssl-certificates=my-cert

# RDS SSL 강제
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --ca-certificate-identifier rds-ca-2019

# Cloud SQL SSL 강제
gcloud sql instances patch my-database --require-ssl
```

---

## 7. 취약점 탐지

### 7.1 AWS Inspector

```bash
# Inspector v2 활성화 (계정 수준)
aws inspector2 enable \
    --resource-types EC2 ECR

# 스캔 결과 조회
aws inspector2 list-findings \
    --filter-criteria '{
        "findingStatus": [{"comparison": "EQUALS", "value": "ACTIVE"}],
        "severity": [{"comparison": "EQUALS", "value": "HIGH"}]
    }'
```

### 7.2 GCP Security Command Center

```bash
# 조직 수준 활성화 필요 (Console에서)

# 발견 항목 조회
gcloud scc findings list ORGANIZATION_ID \
    --source=SOURCE_ID \
    --filter="state=\"ACTIVE\""
```

---

## 8. 위협 탐지

### 8.1 AWS GuardDuty

```bash
# GuardDuty 활성화
aws guardduty create-detector --enable

# 결과 조회
aws guardduty list-findings --detector-id DETECTOR_ID

aws guardduty get-findings \
    --detector-id DETECTOR_ID \
    --finding-ids FINDING_ID

# 신뢰할 수 있는 IP 목록 추가
aws guardduty create-ip-set \
    --detector-id DETECTOR_ID \
    --name "Trusted IPs" \
    --format TXT \
    --location s3://my-bucket/trusted-ips.txt \
    --activate
```

### 8.2 GCP Security Command Center

```bash
# 위협 탐지 (Premium 필요)
# Event Threat Detection
# Container Threat Detection
# Virtual Machine Threat Detection

# 조직 정책 위반 확인
gcloud scc findings list ORGANIZATION_ID \
    --source=SECURITY_HEALTH_ANALYTICS \
    --filter="category=\"PUBLIC_BUCKET_ACL\""
```

---

## 9. 감사 로깅

### 9.1 AWS CloudTrail

```bash
# 트레일 생성
aws cloudtrail create-trail \
    --name my-trail \
    --s3-bucket-name my-log-bucket \
    --is-multi-region-trail \
    --enable-log-file-validation

# 로깅 시작
aws cloudtrail start-logging --name my-trail

# 이벤트 조회
aws cloudtrail lookup-events \
    --lookup-attributes AttributeKey=EventName,AttributeValue=ConsoleLogin \
    --start-time 2024-01-01T00:00:00Z
```

### 9.2 GCP Cloud Audit Logs

```bash
# 감사 로그는 기본 활성화

# 로그 조회
gcloud logging read 'logName:"cloudaudit.googleapis.com"' \
    --project=PROJECT_ID \
    --limit=10

# Data Access 로그 활성화 (추가 설정 필요)
gcloud projects get-iam-policy PROJECT_ID --format=json > policy.json
# 수정 후
gcloud projects set-iam-policy PROJECT_ID policy.json
```

---

## 10. 보안 체크리스트

### 10.1 계정/IAM
```
□ Root/Owner MFA 활성화
□ 최소 권한 원칙 적용
□ 정기적인 권한 검토
□ 미사용 자격 증명 비활성화
□ 강력한 비밀번호 정책
```

### 10.2 네트워크
```
□ 기본 보안 그룹 규칙 제거
□ 필요한 포트만 개방
□ 프라이빗 서브넷 활용
□ VPC Flow Logs 활성화
□ WAF 적용 (웹 앱)
```

### 10.3 데이터
```
□ 저장 시 암호화 활성화
□ 전송 중 암호화 (HTTPS/TLS)
□ 퍼블릭 액세스 차단
□ 백업 암호화
□ 키 로테이션
```

### 10.4 모니터링
```
□ CloudTrail/Audit Logs 활성화
□ GuardDuty/SCC 활성화
□ 보안 알림 설정
□ 정기적인 취약점 스캔
□ 인시던트 대응 계획
```

---

## 11. 다음 단계

- [15_CLI_and_SDK.md](./15_CLI_and_SDK.md) - CLI/SDK 자동화
- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM 상세

---

## 연습 문제

### 연습 문제 1: 심층 방어(Defense-in-Depth) 계층 매핑

한 회사가 ALB 뒤에 위치한 EC2에서 웹 애플리케이션을 운영하고 있습니다. 아래 각 보안 우려사항에 적합한 AWS 서비스 또는 기능을 매핑하세요.

| 보안 우려사항 | AWS 서비스 / 기능 |
|---|---|
| 인터넷에서 유입되는 SQL 인젝션 공격 차단 | ? |
| 비정상적인 IAM 활동 등 의심스러운 API 호출 감지 | ? |
| 고객 관리형 키로 RDS 데이터베이스를 저장 시 암호화 | ? |
| EC2 인스턴스에 대한 직접적인 인터넷 접근 차단 | ? |
| 30일마다 데이터베이스 비밀번호 자동 교체 | ? |

<details>
<summary>정답 보기</summary>

| 보안 우려사항 | AWS 서비스 / 기능 |
|---|---|
| 인터넷에서 유입되는 SQL 인젝션 공격 차단 | AWS WAF + AWSManagedRulesSQLiRuleSet |
| 비정상적인 IAM 활동 등 의심스러운 API 호출 감지 | Amazon GuardDuty |
| 고객 관리형 키로 RDS 데이터베이스를 저장 시 암호화 | AWS KMS(고객 관리형 키) + RDS 암호화 |
| EC2 인스턴스에 대한 직접적인 인터넷 접근 차단 | 프라이빗 서브넷 (퍼블릭 IP 없음) + 보안 그룹(Security Group) (0.0.0.0/0 인바운드 없음) |
| 30일마다 데이터베이스 비밀번호 자동 교체 | 자동 로테이션이 설정된 AWS Secrets Manager |

심층 방어(Defense-in-Depth)란 각 계층이 서로 다른 위협 벡터를 담당하는 것을 의미합니다: 애플리케이션 엣지의 WAF, 계정 수준의 GuardDuty, 저장 데이터를 위한 KMS, 네트워크 접근을 위한 VPC/보안 그룹, 자격 증명 위생을 위한 Secrets Manager.

</details>

---

### 연습 문제 2: KMS 암호화 워크플로우

민감한 설정 파일을 S3에 저장하기 전에 암호화해야 합니다. 다음을 수행하는 AWS CLI 명령어를 작성하세요:
1. `alias/config-key` 별칭으로 고객 관리형 KMS 키 생성
2. 해당 키를 사용하여 `config.json` 암호화
3. S3 버킷 `my-config-bucket`이 `alias/config-key`를 사용하여 기본적으로 SSE-KMS를 적용하도록 설정

<details>
<summary>정답 보기</summary>

```bash
# 1단계: 고객 관리형 키 생성
aws kms create-key \
    --description "Config file encryption key" \
    --key-usage ENCRYPT_DECRYPT \
    --origin AWS_KMS

# 출력의 KeyId를 메모한 후 별칭 생성
aws kms create-alias \
    --alias-name alias/config-key \
    --target-key-id <위에서-확인한-KeyId>

# 2단계: 파일 암호화
aws kms encrypt \
    --key-id alias/config-key \
    --plaintext fileb://config.json \
    --output text \
    --query CiphertextBlob | base64 --decode > config.json.enc

# 3단계: 버킷 기본 암호화 설정
aws s3api put-bucket-encryption \
    --bucket my-config-bucket \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "aws:kms",
                "KMSMasterKeyID": "alias/config-key"
            }
        }]
    }'
```

핵심 포인트:
- 고객 관리형 키(CMK)는 키 정책, 로테이션, 삭제에 대한 제어권을 부여합니다
- SSE-KMS는 업로드 시 객체를 자동으로 암호화하므로 각 파일을 수동으로 암호화할 필요가 없습니다
- 수동 `kms encrypt`의 경우 출력이 base64로 인코딩되어 있으므로 저장 전에 디코딩해야 합니다

</details>

---

### 연습 문제 3: WAF 규칙 설계

이커머스 API 엔드포인트 `POST /api/checkout`에 자동화된 남용이 발생하고 있습니다 — 봇이 다양한 IP에서 분당 수천 건의 가짜 주문을 제출하고 있습니다. 이를 완화하기 위한 GCP Cloud Armor 정책을 설계하세요. 관련 `gcloud` 명령어를 작성하세요.

<details>
<summary>정답 보기</summary>

```bash
# 보안 정책 생성
gcloud compute security-policies create checkout-protection \
    --description="Protect checkout endpoint from abuse"

# 규칙 1: 알려진 악의적인 패턴 차단 (SQL 인젝션)
gcloud compute security-policies rules create 1000 \
    --security-policy=checkout-protection \
    --expression="evaluatePreconfiguredWaf('sqli-v33-stable')" \
    --action=deny-403

# 규칙 2: 속도 제한 — IP당 최대 10 요청/분 허용
gcloud compute security-policies rules create 2000 \
    --security-policy=checkout-protection \
    --expression="request.path.matches('/api/checkout')" \
    --action=rate-based-ban \
    --rate-limit-threshold-count=10 \
    --rate-limit-threshold-interval-sec=60 \
    --ban-duration-sec=300

# 백엔드 서비스에 연결
gcloud compute backend-services update checkout-backend \
    --security-policy=checkout-protection \
    --global
```

설계 근거:
- Cloud Armor에서 낮은 우선순위 번호 = 높은 우선순위 (규칙 1000이 2000보다 먼저 평가됨)
- 속도 기반 차단(Rate-based banning)은 임계값을 초과한 IP를 일시적으로 차단하여 모든 사용자를 차단하지 않고 봇 트래픽을 줄입니다
- 더 정교한 봇 완화를 위해서는 Cloud Armor의 봇 관리 기능과 reCAPTCHA 통합을 고려하세요

</details>

---

### 연습 문제 4: Secrets Manager vs KMS — 언제 무엇을 사용할까

개발자가 묻습니다: "Lambda 함수가 사용하는 서드파티 API 키를 저장해야 합니다. AWS Secrets Manager와 AWS KMS 중 어느 것을 사용해야 할까요?"

차이점을 설명하고 올바른 접근 방식을 추천하세요.

<details>
<summary>정답 보기</summary>

**AWS Secrets Manager를 사용**하여 API 키를 저장하세요. 이유는 다음과 같습니다:

| 기능 | AWS Secrets Manager | AWS KMS (직접 사용) |
|---|---|---|
| 목적 | 비밀 값 저장 및 조회 (비밀번호, API 키, 토큰) | 임의 데이터 암호화/복호화; 암호화 키 관리 |
| 비밀 저장 | 예 — 비밀 값을 안전하게 저장 | 아니오 — 키만 관리; 암호문은 직접 저장해야 함 |
| 자동 로테이션 | 예 — Lambda를 통한 네이티브 로테이션 | 아니오 — 직접 로테이션 로직을 구축해야 함 |
| 접근 제어 | 시크릿에 대한 IAM 정책 | 키 정책 + IAM 정책 |
| 비용 | 시크릿당 $0.40/월 + API 호출 | 키당 $1/월 + API 호출 |

**권장 접근 방식:**
```bash
# API 키 저장
aws secretsmanager create-secret \
    --name /lambda/third-party-api-key \
    --secret-string '{"api_key": "sk-abc123..."}'

# Lambda 함수가 런타임에 조회
```

```python
import boto3, json

def get_api_key():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='/lambda/third-party-api-key')
    return json.loads(response['SecretString'])['api_key']
```

참고: Secrets Manager는 내부적으로 KMS를 사용하여 저장된 시크릿을 암호화합니다. 둘 중 하나를 선택하는 것이 아닙니다 — Secrets Manager는 KMS를 래핑하여 더 높은 수준의 시크릿 저장 서비스를 제공합니다. KMS를 직접 사용하는 것은 자체 데이터 블롭을 암호화해야 할 때만 권장됩니다 (예: 데이터베이스 필드의 애플리케이션 수준 암호화).

</details>

---

### 연습 문제 5: 보안 감사 추적(Audit Trail)

보안 팀은 계정의 모든 AWS API 호출을 기록하고, 1년간 보존하며, 삭제로부터 보호할 것을 요구합니다. 또한 누군가 AWS 콘솔에 로그인할 때 알림을 받고 싶습니다. 단계별 절차와 명령어를 작성하세요.

<details>
<summary>정답 보기</summary>

```bash
# 1단계: 로그 저장용 S3 버킷 생성 (버전 관리 적용)
aws s3api create-bucket \
    --bucket my-cloudtrail-logs-$(date +%s) \
    --region us-east-1

# 버전 관리 활성화 (로그 객체의 우발적 삭제 방지)
aws s3api put-bucket-versioning \
    --bucket my-cloudtrail-logs \
    --versioning-configuration Status=Enabled

# 수명 주기 정책 적용: 90일 후 Glacier로 전환, 1년 후 만료
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-cloudtrail-logs \
    --lifecycle-configuration '{
        "Rules": [{
            "Status": "Enabled",
            "Transitions": [{"Days": 90, "StorageClass": "GLACIER"}],
            "Expiration": {"Days": 365},
            "Filter": {"Prefix": ""}
        }]
    }'

# 2단계: 로그 파일 유효성 검사가 포함된 멀티 리전 CloudTrail 생성
aws cloudtrail create-trail \
    --name org-audit-trail \
    --s3-bucket-name my-cloudtrail-logs \
    --is-multi-region-trail \
    --enable-log-file-validation

aws cloudtrail start-logging --name org-audit-trail

# 3단계: 콘솔 로그인에 대한 CloudWatch 지표 필터 생성
aws logs put-metric-filter \
    --log-group-name CloudTrail/logs \
    --filter-name ConsoleLoginFilter \
    --filter-pattern '{ $.eventName = "ConsoleLogin" }' \
    --metric-transformations metricName=ConsoleLoginCount,metricNamespace=Security,metricValue=1

# 알람 생성
aws cloudwatch put-metric-alarm \
    --alarm-name ConsoleLoginAlert \
    --metric-name ConsoleLoginCount \
    --namespace Security \
    --statistic Sum \
    --period 300 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:...:security-alerts
```

핵심 포인트:
- `--is-multi-region-trail`은 단일 트레일로 모든 리전의 API 활동을 캡처합니다
- `--enable-log-file-validation`은 무결성 검증을 위한 다이제스트 파일을 생성합니다
- 더 엄격한 불변성이 필요한 경우 S3 Object Lock(WORM 모드)을 사용하세요 (컴플라이언스 용도)
- CloudTrail 로그 + CloudWatch 지표 필터 = 준실시간 보안 알림

</details>

---

## 참고 자료

- [AWS Security Best Practices](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/)
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices)
- [AWS WAF](https://docs.aws.amazon.com/waf/)
- [GCP Cloud Armor](https://cloud.google.com/armor/docs)
