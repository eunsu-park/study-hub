# 모니터링, 로깅 & 비용 관리

**이전**: [Infrastructure as Code](./16_Infrastructure_as_Code.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 모니터링(monitoring), 로깅(logging), 알림(alerting)이 프로덕션 클라우드 시스템에 필수적인 이유를 설명할 수 있습니다
2. AWS CloudWatch와 GCP Cloud Monitoring/Logging 서비스 기능을 비교할 수 있습니다
3. 주요 애플리케이션 지표에 대한 커스텀 메트릭(custom metrics), 대시보드(dashboard), 알람(alarm)을 구성할 수 있습니다
4. 중앙 집중식 로그 수집과 구조화된 로그 쿼리를 설정할 수 있습니다
5. 분산 추적(distributed tracing)을 구현하여 마이크로서비스 아키텍처의 지연(latency)을 진단할 수 있습니다
6. 비용 관리 도구(Cost Explorer, Budgets)를 사용해 클라우드 지출을 추적하고 최적화할 수 있습니다
7. 팀과 프로젝트별 비용 배분을 위한 태그(tagging) 전략을 설계할 수 있습니다

---

클라우드 인프라를 구축하는 것은 절반에 불과합니다. 안정적으로 운영하려면 성능, 오류, 비용에 대한 지속적인 가시성(visibility)이 필요합니다. 모니터링 없이는 불만을 가진 사용자를 통해 장애를 알게 되고, 로깅 없이는 원인을 파악하지 못한 채 디버깅해야 합니다. 비용 관리 없이는 잊혀진 리소스가 예산을 조용히 소모할 수 있습니다. 이 레슨에서는 클라우드 환경을 건강하고 예측 가능하게 유지하는 관찰 가능성(observability)과 재정 통제(financial controls)를 다룹니다.

## 1. 모니터링 개요

### 1.1 모니터링이 필요한 이유

- 시스템 가용성 확보
- 성능 문제 조기 발견
- 용량 계획
- 비용 최적화
- 보안 이상 탐지

### 1.2 서비스 매핑

| 기능 | AWS | GCP |
|------|-----|-----|
| 메트릭 모니터링 | CloudWatch | Cloud Monitoring |
| 로그 수집 | CloudWatch Logs | Cloud Logging |
| 추적 | X-Ray | Cloud Trace |
| 대시보드 | CloudWatch Dashboards | Cloud Monitoring Dashboards |
| 알림 | CloudWatch Alarms + SNS | Alerting Policies |
| 비용 관리 | Cost Explorer, Budgets | Billing, Budgets |

---

## 2. AWS CloudWatch

### 2.1 메트릭

```bash
# EC2 메트릭 조회
aws cloudwatch list-metrics --namespace AWS/EC2

# 메트릭 데이터 조회
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-01T23:59:59Z \
    --period 300 \
    --statistics Average

# 커스텀 메트릭 발행
aws cloudwatch put-metric-data \
    --namespace MyApp \
    --metric-name RequestCount \
    --value 100 \
    --unit Count \
    --dimensions Environment=Production
```

**주요 메트릭:**

| 서비스 | 메트릭 | 설명 |
|--------|--------|------|
| EC2 | CPUUtilization | CPU 사용률 |
| EC2 | NetworkIn/Out | 네트워크 트래픽 |
| RDS | DatabaseConnections | DB 연결 수 |
| RDS | FreeStorageSpace | 남은 스토리지 |
| ALB | RequestCount | 요청 수 |
| ALB | TargetResponseTime | 응답 시간 |
| Lambda | Invocations | 호출 수 |
| Lambda | Duration | 실행 시간 |

### 2.2 알람

```bash
# CPU 알람 생성
aws cloudwatch put-metric-alarm \
    --alarm-name high-cpu \
    --alarm-description "CPU over 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:ap-northeast-2:123456789012:alerts

# 알람 목록
aws cloudwatch describe-alarms

# 알람 상태 확인
aws cloudwatch describe-alarm-history \
    --alarm-name high-cpu
```

### 2.3 대시보드

```bash
# 대시보드 생성
aws cloudwatch put-dashboard \
    --dashboard-name MyDashboard \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/EC2", "CPUUtilization", "InstanceId", "i-xxx"]
                    ],
                    "title": "EC2 CPU",
                    "period": 300
                }
            }
        ]
    }'
```

---

## 3. AWS CloudWatch Logs

### 3.1 로그 그룹 관리

```bash
# 로그 그룹 생성
aws logs create-log-group --log-group-name /myapp/production

# 보존 기간 설정
aws logs put-retention-policy \
    --log-group-name /myapp/production \
    --retention-in-days 30

# 로그 스트림 조회
aws logs describe-log-streams --log-group-name /myapp/production

# 로그 조회
aws logs filter-log-events \
    --log-group-name /myapp/production \
    --filter-pattern "ERROR" \
    --start-time 1704067200000 \
    --end-time 1704153600000
```

### 3.2 로그 인사이트

```bash
# 로그 쿼리 실행
aws logs start-query \
    --log-group-name /myapp/production \
    --start-time 1704067200 \
    --end-time 1704153600 \
    --query-string 'fields @timestamp, @message
        | filter @message like /ERROR/
        | sort @timestamp desc
        | limit 20'

# 쿼리 결과 조회
aws logs get-query-results --query-id QUERY_ID
```

### 3.3 EC2에서 로그 전송

```bash
# CloudWatch Agent 설치 (Amazon Linux)
sudo yum install -y amazon-cloudwatch-agent

# 설정 파일
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/myapp/*.log",
                        "log_group_name": "/myapp/production",
                        "log_stream_name": "{instance_id}"
                    }
                ]
            }
        }
    }
}
EOF

# Agent 시작
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s
```

---

## 4. GCP Cloud Monitoring

### 4.1 메트릭

```bash
# 메트릭 목록 조회
gcloud monitoring metrics list --filter="metric.type:compute.googleapis.com"

# 메트릭 데이터 조회 (gcloud에서는 제한적, API/콘솔 권장)
gcloud monitoring metrics read \
    "compute.googleapis.com/instance/cpu/utilization" \
    --project=PROJECT_ID
```

**주요 메트릭:**

| 서비스 | 메트릭 | 설명 |
|--------|--------|------|
| Compute | cpu/utilization | CPU 사용률 |
| Compute | network/received_bytes | 수신 트래픽 |
| Cloud SQL | database/disk/utilization | 디스크 사용률 |
| Cloud Run | request_count | 요청 수 |
| GKE | node/cpu/utilization | 노드 CPU |

### 4.2 알림 정책

```bash
# 알림 채널 생성 (이메일)
gcloud alpha monitoring channels create \
    --display-name="Email Alerts" \
    --type=email \
    --channel-labels=email_address=admin@example.com

# 알림 정책 생성
gcloud alpha monitoring policies create \
    --display-name="High CPU Alert" \
    --condition-display-name="CPU > 80%" \
    --condition-filter='metric.type="compute.googleapis.com/instance/cpu/utilization"' \
    --condition-threshold-value=0.8 \
    --condition-threshold-comparison=COMPARISON_GT \
    --condition-threshold-duration=300s \
    --notification-channels=projects/PROJECT/notificationChannels/CHANNEL_ID
```

---

## 5. GCP Cloud Logging

### 5.1 로그 조회

```bash
# 로그 조회
gcloud logging read 'resource.type="gce_instance"' \
    --limit=10 \
    --format=json

# 에러 로그만
gcloud logging read 'severity>=ERROR' \
    --limit=20

# 특정 시간대
gcloud logging read 'timestamp>="2024-01-01T00:00:00Z"' \
    --limit=100

# 로그 싱크 생성 (Cloud Storage로 내보내기)
gcloud logging sinks create my-sink \
    storage.googleapis.com/my-log-bucket \
    --log-filter='resource.type="gce_instance"'
```

### 5.2 로그 기반 메트릭

```bash
# 에러 수 메트릭 생성
gcloud logging metrics create error-count \
    --description="Count of errors" \
    --log-filter='severity>=ERROR'

# 메트릭 목록
gcloud logging metrics list
```

---

## 6. 비용 관리

### 6.1 AWS Cost Explorer

```bash
# 월별 비용 조회
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE

# 서비스별 비용
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics UnblendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --output table
```

### 6.2 AWS Budgets

```bash
# 월 예산 생성
aws budgets create-budget \
    --account-id 123456789012 \
    --budget '{
        "BudgetName": "Monthly-100USD",
        "BudgetLimit": {"Amount": "100", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80,
                "ThresholdType": "PERCENTAGE"
            },
            "Subscribers": [
                {"SubscriptionType": "EMAIL", "Address": "admin@example.com"}
            ]
        }
    ]'

# 예산 목록
aws budgets describe-budgets --account-id 123456789012
```

### 6.3 GCP Billing

```bash
# 빌링 계정 조회
gcloud billing accounts list

# 프로젝트 빌링 연결
gcloud billing projects link PROJECT_ID \
    --billing-account=BILLING_ACCOUNT_ID

# 예산 생성
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Monthly Budget" \
    --budget-amount=100USD \
    --threshold-rule=percent=0.8,basis=CURRENT_SPEND \
    --all-updates-rule-pubsub-topic=projects/PROJECT/topics/budget-alerts
```

---

## 7. 비용 최적화 전략

### 7.1 컴퓨팅 최적화

| 전략 | AWS | GCP |
|------|-----|-----|
| 예약 인스턴스 | Reserved Instances | Committed Use |
| 스팟/선점형 | Spot Instances | Spot/Preemptible VMs |
| 오토스케일링 | Auto Scaling | Managed Instance Groups |
| 적정 사이징 | AWS Compute Optimizer | Recommender |

```bash
# AWS 권장 사항 조회
aws compute-optimizer get-ec2-instance-recommendations

# GCP 권장 사항 조회
gcloud recommender recommendations list \
    --project=PROJECT_ID \
    --location=global \
    --recommender=google.compute.instance.MachineTypeRecommender
```

### 7.2 스토리지 최적화

```bash
# S3 스토리지 클래스 전환
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration '{
        "Rules": [{
            "ID": "Archive old data",
            "Status": "Enabled",
            "Transitions": [
                {"Days": 30, "StorageClass": "STANDARD_IA"},
                {"Days": 90, "StorageClass": "GLACIER"}
            ]
        }]
    }'

# GCP 수명 주기 정책
gsutil lifecycle set lifecycle.json gs://my-bucket
```

### 7.3 비용 절감 체크리스트

```
□ 미사용 리소스 정리
  - 중지된 인스턴스 (스토리지 비용 계속 발생)
  - 연결되지 않은 EBS/PD 볼륨
  - 오래된 스냅샷
  - 미사용 Elastic IP / 정적 IP

□ 적정 사이징
  - 인스턴스 사용률 분석
  - 오버프로비저닝 확인
  - Rightsizing 권장사항 적용

□ 예약 용량
  - 안정적 워크로드에 예약 인스턴스
  - 1년/3년 약정 검토

□ 스팟/선점형 활용
  - 배치 작업, 개발 환경
  - 중단 허용 워크로드

□ 스토리지 최적화
  - 수명 주기 정책 적용
  - 적절한 스토리지 클래스
  - 불필요한 데이터 정리

□ 네트워크 비용
  - 같은 AZ/리전 내 통신
  - CDN 활용
  - NAT Gateway 트래픽 최적화
```

---

## 8. 태그 기반 비용 추적

### 8.1 태그 전략

```hcl
# Terraform 예시
locals {
  common_tags = {
    Environment = "production"
    Project     = "myapp"
    CostCenter  = "engineering"
    Owner       = "team-a"
    ManagedBy   = "terraform"
  }
}

resource "aws_instance" "web" {
  # ...
  tags = local.common_tags
}
```

### 8.2 비용 할당 태그

```bash
# AWS 비용 할당 태그 활성화 (Billing Console에서)

# 태그별 비용 조회
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=TAG,Key=Project

# GCP 라벨별 비용 (BigQuery 내보내기 필요)
SELECT
  labels.key,
  labels.value,
  SUM(cost) as total_cost
FROM `billing_export.gcp_billing_export_v1_*`
CROSS JOIN UNNEST(labels) as labels
GROUP BY 1, 2
ORDER BY total_cost DESC
```

---

## 9. 대시보드 예시

### 9.1 운영 대시보드 구성

```
┌──────────────────────────────────────────────────────────────┐
│  운영 대시보드                                               │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   CPU 사용률    │  │  메모리 사용률  │  │  요청 수     │ │
│  │   [그래프]      │  │   [그래프]      │  │  [그래프]    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   응답 시간     │  │   에러율        │  │  활성 연결   │ │
│  │   [그래프]      │  │   [그래프]      │  │  [그래프]    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   최근 알람 / 인시던트                                  │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   비용 요약 (이번 달)                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. 알림 설정 권장사항

### 10.1 필수 알림

| 카테고리 | 조건 | 긴급도 |
|----------|------|--------|
| CPU | > 80% (5분) | 중 |
| CPU | > 95% (2분) | 높 |
| 메모리 | > 85% | 중 |
| 디스크 | > 80% | 중 |
| 디스크 | > 90% | 높 |
| 헬스체크 | 실패 | 높 |
| 에러율 | > 1% | 중 |
| 에러율 | > 5% | 높 |
| 응답 시간 | > 2초 | 중 |
| 비용 | > 80% 예산 | 중 |

### 10.2 알림 채널

```bash
# AWS SNS 토픽 생성
aws sns create-topic --name alerts

# 이메일 구독
aws sns subscribe \
    --topic-arn arn:aws:sns:...:alerts \
    --protocol email \
    --notification-endpoint admin@example.com

# Slack 웹훅 (Lambda 통해)
# PagerDuty, Opsgenie 등 연동
```

---

## 11. 다음 단계

- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC Flow Logs
- [14_Security_Services.md](./14_Security_Services.md) - 보안 모니터링

---

## 연습 문제

### 연습 문제 1: 모니터링 서비스 매핑

새 애플리케이션의 가관측성(Observability) 스택을 설계하고 있습니다. 각 요구사항에 대해 올바른 AWS와 GCP 서비스를 식별하세요.

| 요구사항 | AWS 서비스 | GCP 서비스 |
|---|---|---|
| Lambda 함수 실행 시간 추적 | ? | ? |
| 애플리케이션 로그 출력 저장 및 검색 | ? | ? |
| 디스크 사용률이 90%를 초과할 때 자동 알림 | ? | ? |
| 병목 지점을 찾기 위해 여러 마이크로서비스를 통한 요청 추적 | ? | ? |
| SQL 유사 구문으로 지난 24시간의 에러 로그 쿼리 | ? | ? |

<details>
<summary>정답 보기</summary>

| 요구사항 | AWS 서비스 | GCP 서비스 |
|---|---|---|
| Lambda 함수 실행 시간 추적 | CloudWatch 지표 (`Lambda/Duration`) | Cloud Monitoring (`cloudfunctions.googleapis.com/function/execution_times`) |
| 애플리케이션 로그 출력 저장 및 검색 | CloudWatch Logs | Cloud Logging |
| 디스크 사용률이 90%를 초과할 때 자동 알림 | CloudWatch 알람 + SNS | Cloud Monitoring 알림 정책 + 알림 채널 |
| 병목 지점을 찾기 위해 여러 마이크로서비스를 통한 요청 추적 | AWS X-Ray | Cloud Trace |
| SQL 유사 구문으로 지난 24시간의 에러 로그 쿼리 | CloudWatch Logs Insights | Cloud Logging (Log Analytics / BigQuery 내보내기 포함) |

참고: GCP 로그 쿼리에서 Cloud Logging은 자체 필터 구문을 사용하지만(SQL 아님), SQL 기반 분석을 위해 BigQuery로 내보낼 수 있습니다. CloudWatch Logs Insights는 SQL과 유사한 자체 쿼리 언어를 갖추고 있습니다.

</details>

---

### 연습 문제 2: CloudWatch 알람 설계

프로덕션 웹 애플리케이션에 다음과 같은 SLO(서비스 수준 목표) 요구사항이 있습니다:
- 가용성: 5초 이내에 헬스 체크에 응답
- 지연 시간: p95 응답 시간이 2초 미만이어야 함
- 에러율: HTTP 5xx 에러가 전체 요청의 1% 미만이어야 함

이 세 가지 SLO 각각에 대한 알람을 생성하는 CloudWatch CLI 명령어를 작성하세요. SNS 토픽 `arn:aws:sns:ap-northeast-2:123456789012:prod-alerts`로 알림을 보내야 합니다.

<details>
<summary>정답 보기</summary>

```bash
# 1. 헬스 체크 가용성: ALB 비정상 호스트 수 알람
aws cloudwatch put-metric-alarm \
    --alarm-name "prod-unhealthy-hosts" \
    --alarm-description "ALB에 비정상 타겟이 있음" \
    --metric-name UnHealthyHostCount \
    --namespace AWS/ApplicationELB \
    --dimensions Name=LoadBalancer,Value=app/my-alb/1234567890abcdef \
    --statistic Average \
    --period 60 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:ap-northeast-2:123456789012:prod-alerts

# 2. 지연 시간: p95 타겟 응답 시간 > 2초
aws cloudwatch put-metric-alarm \
    --alarm-name "prod-high-latency-p95" \
    --alarm-description "p95 지연 시간이 2초를 초과함" \
    --metric-name TargetResponseTime \
    --namespace AWS/ApplicationELB \
    --dimensions Name=LoadBalancer,Value=app/my-alb/1234567890abcdef \
    --extended-statistic p95 \
    --period 300 \
    --threshold 2 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 3 \
    --alarm-actions arn:aws:sns:ap-northeast-2:123456789012:prod-alerts

# 3. 에러율: 5xx 에러 > 전체 요청의 1%
# 비율을 계산하기 위해 지표 수식(Metric Math) 알람 사용
aws cloudwatch put-metric-alarm \
    --alarm-name "prod-high-5xx-rate" \
    --alarm-description "5xx 에러율이 1%를 초과함" \
    --metrics '[
        {"Id":"e1","Expression":"m2/m1*100","Label":"5xx Rate %"},
        {"Id":"m1","MetricStat":{"Metric":{"Namespace":"AWS/ApplicationELB","MetricName":"RequestCount","Dimensions":[{"Name":"LoadBalancer","Value":"app/my-alb/1234567890abcdef"}]},"Period":300,"Stat":"Sum"},"ReturnData":false},
        {"Id":"m2","MetricStat":{"Metric":{"Namespace":"AWS/ApplicationELB","MetricName":"HTTPCode_Target_5XX_Count","Dimensions":[{"Name":"LoadBalancer","Value":"app/my-alb/1234567890abcdef"}]},"Period":300,"Stat":"Sum"},"ReturnData":false}
    ]' \
    --comparison-operator GreaterThanThreshold \
    --threshold 1 \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:ap-northeast-2:123456789012:prod-alerts
```

참고: 에러율 알람은 지표 수식(`--metrics`와 `Expression` 사용)을 사용하여 백분율을 계산합니다. 이렇게 하면 (트래픽이 낮을 때 알람이 발생하는) 원시 카운트로 알람을 설정하는 것을 피하고 비율을 측정할 수 있습니다.

</details>

---

### 연습 문제 3: CloudWatch Logs Insights 로그 쿼리

애플리케이션이 구조화된 JSON을 로깅합니다. 고객이 2024-03-15 UTC 14:30경에 오류를 보고했습니다. `/myapp/production` 로그 그룹에서 `payment` 문자열이 포함된 가장 최근 `ERROR` 수준의 로그 항목 20개를 찾는 Logs Insights 쿼리를 작성하고, 타임스탬프, 요청 ID, 에러 메시지를 표시하세요.

<details>
<summary>정답 보기</summary>

```bash
# 쿼리 시작
aws logs start-query \
    --log-group-name /myapp/production \
    --start-time $(date -d "2024-03-15T14:00:00Z" +%s) \
    --end-time $(date -d "2024-03-15T15:00:00Z" +%s) \
    --query-string '
        fields @timestamp, requestId, message
        | filter level = "ERROR" and message like /payment/
        | sort @timestamp desc
        | limit 20
    '

# 출력에서 queryId를 메모한 후 결과 조회:
aws logs get-query-results --query-id <QUERY_ID>
```

**Logs Insights 쿼리 구문 설명:**
- `fields` — 표시할 필드 선택
- `filter` — 행 필터링 (SQL WHERE 구문과 유사); `and`, `or`, `not`, `like /regex/` 지원
- `sort` — 결과 정렬; `@timestamp`는 내장 필드
- `limit` — 반환되는 결과 수 제한

**일반적인 Logs Insights 패턴:**
```
# 유형별 에러 수 계산
fields errorType
| filter level = "ERROR"
| stats count(*) as errorCount by errorType
| sort errorCount desc

# 엔드포인트별 95번째 백분위 지연 시간
fields endpoint, duration
| stats pct(duration, 95) as p95 by endpoint
| sort p95 desc
```

</details>

---

### 연습 문제 4: AWS 예산(Budget) 알림

팀의 월간 AWS 예산은 $500입니다. 실제 지출이 60%, 80%, 100%에 도달할 때 이메일 알림을 받고 싶고, 예상 월말 지출이 예산의 110%를 초과할 것으로 예측될 때도 예측 알림을 받고 싶습니다. AWS CLI 명령어를 작성하세요.

<details>
<summary>정답 보기</summary>

```bash
aws budgets create-budget \
    --account-id 123456789012 \
    --budget '{
        "BudgetName": "Monthly-500USD",
        "BudgetLimit": {
            "Amount": "500",
            "Unit": "USD"
        },
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 60,
                "ThresholdType": "PERCENTAGE"
            },
            "Subscribers": [
                {"SubscriptionType": "EMAIL", "Address": "team@example.com"}
            ]
        },
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80,
                "ThresholdType": "PERCENTAGE"
            },
            "Subscribers": [
                {"SubscriptionType": "EMAIL", "Address": "team@example.com"}
            ]
        },
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100,
                "ThresholdType": "PERCENTAGE"
            },
            "Subscribers": [
                {"SubscriptionType": "EMAIL", "Address": "team@example.com"}
            ]
        },
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 110,
                "ThresholdType": "PERCENTAGE"
            },
            "Subscribers": [
                {"SubscriptionType": "EMAIL", "Address": "team@example.com"}
            ]
        }
    ]'
```

핵심 포인트:
- `NotificationType: "ACTUAL"` — 실제 요금이 임계값을 초과할 때 트리거
- `NotificationType: "FORECASTED"` — AWS가 월말 지출이 임계값을 초과할 것으로 예측할 때 트리거 — 실제로 초과하기 전에 조기 경고를 제공
- 예산당 최대 5개의 알림, 각각 최대 10명의 구독자
- 예산은 SNS 토픽을 트리거할 수도 있음 (dev 인스턴스 종료 등 자동화된 수정 조치용)

</details>

---

### 연습 문제 5: 비용 최적화 분석

AWS Cost Explorer에 따르면 지난 달 청구서는 $1,200으로, 예상 $600보다 훨씬 높습니다. 내역은 다음과 같습니다:
- EC2: $800 (예상 ~$200)
- S3: $150 (예상 ~$50)
- 데이터 전송: $200 (예상 ~$30)

가장 가능성 높은 원인과 각각을 조사하는 데 사용할 CLI 명령어 또는 콘솔 작업을 나열하세요.

<details>
<summary>정답 보기</summary>

**EC2 ($800 vs $200 예상 — 4배 초과):**

가능성 높은 원인:
1. 잊혀진 실행 중인 인스턴스 (특히 대형 인스턴스 유형)
2. 예약 인스턴스(Reserved Instances) 또는 절감형 플랜(Savings Plans)을 사용하지 않는 인스턴스
3. 잘못된 인스턴스 유형 (예: 누군가 실수로 `m5.4xlarge` 시작)

조사:
```bash
# 인스턴스 유형 및 시작 시간이 포함된 모든 실행 중인 인스턴스 나열
aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime,Tags[?Key==`Name`].Value|[0]]' \
    --output table

# 미사용 예약 인스턴스 확인
aws ec2 describe-reserved-instances \
    --filters "Name=state,Values=active"

# 사이즈 조정 권장사항 가져오기
aws compute-optimizer get-ec2-instance-recommendations \
    --query 'instanceRecommendations[?finding==`OVER_PROVISIONED`]'
```

**S3 ($150 vs $50 — 3배 초과):**

가능성 높은 원인:
1. STANDARD_IA 또는 GLACIER에 있어야 할 객체가 STANDARD 클래스에 저장됨
2. 버킷 버전 관리가 활성화되어 이전 버전이 많이 쌓임
3. 완료되지 않은 멀티파트 업로드 미정리

조사:
```bash
# 버킷별 스토리지 클래스 분포 확인
aws s3api list-buckets --query 'Buckets[*].Name' --output text | \
    xargs -I {} aws s3api list-objects-v2 --bucket {} \
    --query 'Contents[*].[StorageClass]' --output text | sort | uniq -c

# 수명 주기 정책이 적용되어 있는지 확인
aws s3api get-bucket-lifecycle-configuration --bucket my-bucket
```

**데이터 전송 ($200 vs $30 — 6.7배 초과):**

가능성 높은 원인:
1. AWS에서 인터넷으로 데이터 전송 (가장 비쌈: ~$0.09/GB)
2. 다른 AZ 간 트래픽 (다른 AZ의 EC2 <-> RDS)
3. NAT 게이트웨이 트래픽 (NAT를 통해 처리된 데이터: ~$0.045/GB)

조사:
```bash
# 트래픽 패턴에 대한 VPC Flow Logs 확인
aws logs start-query \
    --log-group-name VPCFlowLogs \
    --start-time ... --end-time ... \
    --query-string 'stats sum(bytes) as total_bytes by dstAddr | sort total_bytes desc | limit 20'

# 특정 전송 유형을 식별하기 위해 사용 유형별 Cost Explorer 검토
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=USAGE_TYPE
```

일반적인 비용 위생 체크리스트:
- AWS 비용 이상 감지(Cost Anomaly Detection) 활성화 (예상치 못한 지출 급증을 자동으로 알림)
- 비용 귀속을 위해 모든 리소스에 `Project`, `Environment`, `Owner` 태그 지정
- 프로덕션 워크로드를 시작한 후가 아니라 전에 예산 알림 설정

</details>

---

## 참고 자료

- [AWS CloudWatch Documentation](https://docs.aws.amazon.com/cloudwatch/)
- [AWS Cost Management](https://docs.aws.amazon.com/cost-management/)
- [GCP Cloud Monitoring](https://cloud.google.com/monitoring/docs)
- [GCP Billing](https://cloud.google.com/billing/docs)
