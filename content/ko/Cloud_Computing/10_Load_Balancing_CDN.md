# 로드밸런싱 & CDN

**이전**: [VPC](./09_Virtual_Private_Cloud.md) | **다음**: [관리형 관계형 데이터베이스](./11_Managed_Relational_DB.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 고가용성(High Availability)과 확장성(Scalability) 달성에서 로드밸런서의 역할을 설명할 수 있습니다
2. 레이어 4(Layer 4, TCP/UDP) 로드밸런서와 레이어 7(Layer 7, HTTP/HTTPS) 로드밸런서의 차이를 구분할 수 있습니다
3. AWS ELB 유형(ALB, NLB, CLB)과 GCP 로드밸런싱 옵션을 비교할 수 있습니다
4. 자동 장애 조치(Automatic Failover)를 위한 헬스 체크(Health Check)와 대상 그룹(Target Group)을 구성할 수 있습니다
5. CDN(CloudFront, Cloud CDN)이 엣지(Edge)에서 콘텐츠를 캐싱하고 전달하는 방식을 설명할 수 있습니다
6. 로드밸런서가 여러 가용 영역(Zone)에 걸쳐 트래픽을 분산하는 다중 계층 아키텍처를 설계할 수 있습니다

---

애플리케이션이 단일 서버를 넘어 성장함에 따라, 여러 인스턴스에 트래픽을 분산하는 것이 안정성과 성능을 위해 필수적이 됩니다. 로드밸런서는 특정 서버가 병목 지점이 되는 것을 방지하고, CDN은 최종 사용자에게 더 가까운 곳에 콘텐츠를 배치하여 지연 시간(Latency)을 줄입니다. 이 두 서비스는 클라우드 애플리케이션을 대규모에서 빠르고 탄력적으로 만드는 전달 계층(Delivery Layer)을 형성합니다.

## 1. 로드밸런싱 개요

### 1.1 로드밸런서란?

로드밸런서는 들어오는 트래픽을 여러 서버에 분산시키는 서비스입니다.

**장점:**
- 고가용성 (장애 서버 자동 제외)
- 확장성 (서버 추가/제거 용이)
- 성능 향상 (부하 분산)
- 보안 (DDoS 완화, SSL 오프로딩)

### 1.2 서비스 비교

| 항목 | AWS | GCP |
|------|-----|-----|
| L7 (HTTP/HTTPS) | ALB | HTTP(S) Load Balancing |
| L4 (TCP/UDP) | NLB | TCP/UDP Load Balancing |
| 클래식 | CLB (레거시) | - |
| 내부 | Internal ALB/NLB | Internal Load Balancing |
| 글로벌 | Global Accelerator | Global Load Balancing |

---

## 2. AWS Elastic Load Balancing

### 2.1 로드밸런서 유형

| 유형 | 계층 | 사용 사례 | 특징 |
|------|------|----------|------|
| **ALB** | L7 | 웹 앱, 마이크로서비스 | 경로/호스트 라우팅, 웹소켓 |
| **NLB** | L4 | 고성능, 정적 IP 필요 | 수백만 RPS, 초저지연 |
| **GWLB** | L3 | 방화벽, IDS/IPS | 투명 게이트웨이 |

### 2.2 ALB (Application Load Balancer)

```bash
# 1. 대상 그룹 생성
aws elbv2 create-target-group \
    --name my-targets \
    --protocol HTTP \
    --port 80 \
    --vpc-id vpc-12345678 \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --target-type instance

# 2. 인스턴스 등록
aws elbv2 register-targets \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx \
    --targets Id=i-12345678 Id=i-87654321

# 3. ALB 생성
aws elbv2 create-load-balancer \
    --name my-alb \
    --subnets subnet-1 subnet-2 \
    --security-groups sg-12345678 \
    --scheme internet-facing \
    --type application

# 4. 리스너 생성
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx
```

**경로 기반 라우팅:**
```bash
# 규칙 추가 (/api/* → API 대상 그룹)
aws elbv2 create-rule \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --priority 10 \
    --conditions Field=path-pattern,Values='/api/*' \
    --actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/api-targets/xxx
```

### 2.3 NLB (Network Load Balancer)

```bash
# NLB 생성 (정적 IP)
aws elbv2 create-load-balancer \
    --name my-nlb \
    --subnets subnet-1 subnet-2 \
    --type network \
    --scheme internet-facing

# TCP 리스너
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/net/my-nlb/xxx \
    --protocol TCP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/tcp-targets/xxx
```

### 2.4 SSL/TLS 설정

```bash
# ACM 인증서 요청
aws acm request-certificate \
    --domain-name example.com \
    --subject-alternative-names "*.example.com" \
    --validation-method DNS

# HTTPS 리스너 추가
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=arn:aws:acm:...:certificate/xxx \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx

# HTTP → HTTPS 리다이렉트
aws elbv2 modify-listener \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --default-actions Type=redirect,RedirectConfig='{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}'
```

---

## 3. GCP Cloud Load Balancing

### 3.1 로드밸런서 유형

| 유형 | 범위 | 계층 | 사용 사례 |
|------|------|------|----------|
| **Global HTTP(S)** | 글로벌 | L7 | 웹 앱, CDN 통합 |
| **Regional HTTP(S)** | 리전 | L7 | 단일 리전 앱 |
| **Global TCP/SSL** | 글로벌 | L4 | TCP 프록시 |
| **Regional TCP/UDP** | 리전 | L4 | 네트워크 LB |
| **Internal HTTP(S)** | 리전 | L7 | 내부 마이크로서비스 |
| **Internal TCP/UDP** | 리전 | L4 | 내부 TCP/UDP |

### 3.2 HTTP(S) Load Balancer

```bash
# 1. 인스턴스 그룹 생성 (비관리형)
gcloud compute instance-groups unmanaged create my-group \
    --zone=asia-northeast3-a

gcloud compute instance-groups unmanaged add-instances my-group \
    --zone=asia-northeast3-a \
    --instances=instance-1,instance-2

# 2. 헬스 체크 생성
gcloud compute health-checks create http my-health-check \
    --port=80 \
    --request-path=/health

# 3. 백엔드 서비스 생성
gcloud compute backend-services create my-backend \
    --protocol=HTTP \
    --health-checks=my-health-check \
    --global

# 4. 인스턴스 그룹을 백엔드에 추가
gcloud compute backend-services add-backend my-backend \
    --instance-group=my-group \
    --instance-group-zone=asia-northeast3-a \
    --global

# 5. URL 맵 생성
gcloud compute url-maps create my-url-map \
    --default-service=my-backend

# 6. 대상 HTTP 프록시 생성
gcloud compute target-http-proxies create my-proxy \
    --url-map=my-url-map

# 7. 전역 전달 규칙 생성
gcloud compute forwarding-rules create my-lb \
    --global \
    --target-http-proxy=my-proxy \
    --ports=80
```

### 3.3 SSL/TLS 설정

```bash
# 1. 관리형 SSL 인증서
gcloud compute ssl-certificates create my-cert \
    --domains=example.com,www.example.com \
    --global

# 2. HTTPS 대상 프록시
gcloud compute target-https-proxies create my-https-proxy \
    --url-map=my-url-map \
    --ssl-certificates=my-cert

# 3. HTTPS 전달 규칙
gcloud compute forwarding-rules create my-https-lb \
    --global \
    --target-https-proxy=my-https-proxy \
    --ports=443

# 4. HTTP → HTTPS 리다이렉트
gcloud compute url-maps import my-url-map --source=- <<EOF
name: my-url-map
defaultUrlRedirect:
  httpsRedirect: true
  redirectResponseCode: MOVED_PERMANENTLY_DEFAULT
EOF
```

### 3.4 경로 기반 라우팅

```bash
# URL 맵에 경로 규칙 추가
gcloud compute url-maps add-path-matcher my-url-map \
    --path-matcher-name=api-matcher \
    --default-service=default-backend \
    --path-rules="/api/*=api-backend,/static/*=static-backend"
```

---

## 4. 헬스 체크

### 4.1 AWS 헬스 체크

```bash
# 대상 그룹 헬스 체크 설정
aws elbv2 modify-target-group \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx \
    --health-check-protocol HTTP \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3

# 대상 헬스 상태 확인
aws elbv2 describe-target-health \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx
```

### 4.2 GCP 헬스 체크

```bash
# HTTP 헬스 체크
gcloud compute health-checks create http my-http-check \
    --port=80 \
    --request-path=/health \
    --check-interval=30s \
    --timeout=5s \
    --healthy-threshold=2 \
    --unhealthy-threshold=3

# TCP 헬스 체크
gcloud compute health-checks create tcp my-tcp-check \
    --port=3306

# 헬스 체크 상태 확인
gcloud compute backend-services get-health my-backend --global
```

---

## 5. Auto Scaling 연동

### 5.1 AWS Auto Scaling Group + ALB

```bash
# 시작 템플릿 생성
aws ec2 create-launch-template \
    --launch-template-name my-template \
    --launch-template-data '{
        "ImageId": "ami-12345678",
        "InstanceType": "t3.micro",
        "SecurityGroupIds": ["sg-12345678"]
    }'

# Auto Scaling Group 생성 (대상 그룹 연결)
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name my-asg \
    --launch-template LaunchTemplateName=my-template,Version='$Latest' \
    --min-size 2 \
    --max-size 10 \
    --desired-capacity 2 \
    --vpc-zone-identifier "subnet-1,subnet-2" \
    --target-group-arns "arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx"

# 스케일링 정책
aws autoscaling put-scaling-policy \
    --auto-scaling-group-name my-asg \
    --policy-name cpu-scaling \
    --policy-type TargetTrackingScaling \
    --target-tracking-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ASGAverageCPUUtilization"
        }
    }'
```

### 5.2 GCP Managed Instance Group + LB

```bash
# 인스턴스 템플릿 생성
gcloud compute instance-templates create my-template \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server

# 관리형 인스턴스 그룹 생성
gcloud compute instance-groups managed create my-mig \
    --template=my-template \
    --size=2 \
    --zone=asia-northeast3-a

# 오토스케일링 설정
gcloud compute instance-groups managed set-autoscaling my-mig \
    --zone=asia-northeast3-a \
    --min-num-replicas=2 \
    --max-num-replicas=10 \
    --target-cpu-utilization=0.7

# 로드밸런서에 연결
gcloud compute backend-services add-backend my-backend \
    --instance-group=my-mig \
    --instance-group-zone=asia-northeast3-a \
    --global
```

---

## 6. CDN (Content Delivery Network)

### 6.1 AWS CloudFront

```bash
# CloudFront 배포 생성 (S3 오리진)
aws cloudfront create-distribution \
    --distribution-config '{
        "CallerReference": "my-distribution-2024",
        "Origins": {
            "Quantity": 1,
            "Items": [{
                "Id": "S3-my-bucket",
                "DomainName": "my-bucket.s3.amazonaws.com",
                "S3OriginConfig": {
                    "OriginAccessIdentity": ""
                }
            }]
        },
        "DefaultCacheBehavior": {
            "TargetOriginId": "S3-my-bucket",
            "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": {
                "Quantity": 2,
                "Items": ["GET", "HEAD"]
            },
            "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
            "Compress": true
        },
        "Enabled": true,
        "DefaultRootObject": "index.html"
    }'

# 캐시 무효화
aws cloudfront create-invalidation \
    --distribution-id EDFDVBD632BHDS5 \
    --paths "/*"
```

**CloudFront + ALB:**
```bash
# ALB를 오리진으로 하는 CloudFront
{
    "Origins": {
        "Items": [{
            "Id": "ALB-origin",
            "DomainName": "my-alb-12345.ap-northeast-2.elb.amazonaws.com",
            "CustomOriginConfig": {
                "HTTPPort": 80,
                "HTTPSPort": 443,
                "OriginProtocolPolicy": "https-only"
            }
        }]
    }
}
```

### 6.2 GCP Cloud CDN

```bash
# 1. 백엔드 서비스에 CDN 활성화
gcloud compute backend-services update my-backend \
    --enable-cdn \
    --global

# 2. Cloud Storage 버킷을 CDN 오리진으로
gcloud compute backend-buckets create my-cdn-bucket \
    --gcs-bucket-name=my-static-bucket \
    --enable-cdn

# 3. URL 맵에 버킷 추가
gcloud compute url-maps add-path-matcher my-url-map \
    --path-matcher-name=static-matcher \
    --default-backend-bucket=my-cdn-bucket \
    --path-rules="/static/*=my-cdn-bucket"

# 4. 캐시 무효화
gcloud compute url-maps invalidate-cdn-cache my-url-map \
    --path="/*"
```

### 6.3 CDN 캐시 정책

**AWS CloudFront 캐시 정책:**
```bash
# 캐시 정책 생성
aws cloudfront create-cache-policy \
    --cache-policy-config '{
        "Name": "MyPolicy",
        "DefaultTTL": 86400,
        "MaxTTL": 31536000,
        "MinTTL": 0,
        "ParametersInCacheKeyAndForwardedToOrigin": {
            "EnableAcceptEncodingGzip": true,
            "HeadersConfig": {"HeaderBehavior": "none"},
            "CookiesConfig": {"CookieBehavior": "none"},
            "QueryStringsConfig": {"QueryStringBehavior": "none"}
        }
    }'
```

**GCP Cloud CDN 캐시 모드:**
```bash
# 캐시 모드 설정
gcloud compute backend-services update my-backend \
    --cache-mode=CACHE_ALL_STATIC \
    --default-ttl=3600 \
    --max-ttl=86400 \
    --global
```

---

## 7. 비용 비교

### 7.1 로드밸런서 비용

| 서비스 | 고정 비용 | 처리 비용 |
|--------|----------|----------|
| AWS ALB | ~$18/월 | $0.008/LCU-시간 |
| AWS NLB | ~$18/월 | $0.006/NLCU-시간 |
| GCP HTTP(S) LB | ~$18/월 | $0.008/GB 처리량 |
| GCP TCP/UDP LB | 리전당 $18/월 | 규칙당 추가 |

### 7.2 CDN 비용

| 서비스 | 데이터 전송 (첫 10TB) |
|--------|---------------------|
| AWS CloudFront | ~$0.085/GB (미국/유럽) |
| GCP Cloud CDN | ~$0.08/GB (미국/유럽) |

---

## 8. 모니터링

### 8.1 AWS CloudWatch 메트릭

```bash
# ALB 메트릭 조회
aws cloudwatch get-metric-statistics \
    --namespace AWS/ApplicationELB \
    --metric-name RequestCount \
    --dimensions Name=LoadBalancer,Value=app/my-alb/xxx \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-01T23:59:59Z \
    --period 300 \
    --statistics Sum

# 주요 메트릭:
# - RequestCount
# - HTTPCode_Target_2XX_Count
# - TargetResponseTime
# - HealthyHostCount
# - UnHealthyHostCount
```

### 8.2 GCP Cloud Monitoring

```bash
# 메트릭 조회
gcloud monitoring metrics list \
    --filter="metric.type:loadbalancing"

# 알림 정책 생성
gcloud alpha monitoring policies create \
    --display-name="High Latency Alert" \
    --condition-display-name="Latency > 1s" \
    --condition-filter='metric.type="loadbalancing.googleapis.com/https/backend_latencies"' \
    --condition-threshold-value=1000 \
    --notification-channels=projects/PROJECT/notificationChannels/xxx
```

---

## 9. 다음 단계

- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - 데이터베이스
- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - 모니터링 상세

---

## 연습 문제

### 연습 문제 1: ALB vs NLB 선택

각 시나리오에 ALB(Application Load Balancer)와 NLB(Network Load Balancer) 중 하나를 선택하고 이유를 설명하세요:

1. 여러 마이크로서비스 라우트가 있는 REST API 서비스: `/api/users/*`는 사용자 서비스로, `/api/orders/*`는 주문 서비스로 이동합니다.
2. 빠른 패킷 전달을 위해 UDP를 사용하고, 방화벽 화이트리스트 등록을 위한 정적 IP 주소가 필요한 실시간 게임 서버
3. SSL 인증서 관리와 쇼핑 카트 상태를 위한 스티키 세션(sticky session)이 필요한 HTTPS 웹 애플리케이션
4. 서브 밀리초(sub-millisecond) 지연 시간이 필요하고 초당 수백만 TCP 연결을 처리하는 금융 트레이딩 플랫폼

<details>
<summary>정답 보기</summary>

1. **ALB** — 경로 기반 라우팅(path-based routing)은 ALB만의 기능입니다. ALB는 HTTP 요청 경로를 검사하여 URL 패턴에 따라 다른 타깃 그룹으로 라우팅합니다(`/api/users/*` vs `/api/orders/*`). NLB는 레이어 4에서 동작하여 HTTP 경로를 검사할 수 없습니다.

2. **NLB** — NLB는 UDP를 지원하며(ALB는 HTTP/HTTPS만), AZ당 정적 IP 주소를 제공하여 방화벽 화이트리스트 등록에 필수적입니다. NLB는 클라이언트 소스 IP를 타깃에 직접 전달하며, 많은 게임 서버에 필요합니다.

3. **ALB** — ALB는 SSL 종료(서버에서 HTTPS 오프로딩), ACM 인증서 관리, 쿠키 기반 어피니티(affinity)를 통한 스티키 세션을 지원합니다. NLB도 TLS 패스스루(passthrough)를 할 수 있지만, ALB의 L7 기능이 웹 애플리케이션에 더 적합합니다.

4. **NLB** — NLB는 초당 수백만 건의 요청과 초저지연(~100μs vs ALB의 ~1ms)을 위해 설계되었습니다. NLB는 최소한의 처리 오버헤드로 레이어 4에서 동작하여 지연 시간에 민감한 금융 애플리케이션에 이상적입니다.

</details>

### 연습 문제 2: 헬스 체크(Health Check) 설정

ALB에 웹 애플리케이션을 실행하는 EC2 인스턴스 타깃 그룹이 있습니다. 애플리케이션에는 정상일 때 HTTP 200, 사용 불가일 때 HTTP 503을 반환하는 `/health` 엔드포인트가 있습니다.

설정할 헬스 체크 구성을 설명하고, 인스턴스가 헬스 체크에 실패할 때 어떤 일이 발생하는지 설명하세요.

<details>
<summary>정답 보기</summary>

**권장 헬스 체크 설정**:

```bash
aws elbv2 modify-target-group \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx \
    --health-check-protocol HTTP \
    --health-check-port 80 \
    --health-check-path /health \
    --health-check-interval-seconds 15 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --matcher HttpCode=200
```

**설정 설명**:
- `--health-check-path /health` — 리다이렉트 로직이 있을 수 있는 루트 경로 대신 전용 헬스 엔드포인트 사용
- `--health-check-interval-seconds 15` — 15초마다 확인 (응답성과 트래픽 오버헤드 사이의 균형)
- `--healthy-threshold-count 2` — 정상으로 표시되려면 2회 연속 성공 필요 (플래핑, flapping 방지)
- `--unhealthy-threshold-count 3` — 제거 전 3회 연속 실패 필요 (일시적 오류로 인한 조기 제거 방지)
- `--matcher HttpCode=200` — HTTP 200만 허용; 503은 비정상 상태를 나타냄

**인스턴스가 헬스 체크에 실패할 때 발생하는 일**:
1. 3번 연속 실패(3 × 15초 = 45초) 후 인스턴스가 `unhealthy`로 표시됩니다.
2. ALB가 즉시 비정상 인스턴스에 새 요청을 보내는 것을 중단합니다.
3. 기존 진행 중인 연결이 드레인(drain)됩니다(커넥션 드레이닝 기간, 기본 300초).
4. Auto Scaling Group과 통합된 경우, ASG가 비정상 인스턴스를 감지하고 종료하여 대체 인스턴스를 시작합니다.
5. 대체 인스턴스가 2번 연속 헬스 체크를 통과하면 트래픽을 받기 시작합니다.

</details>

### 연습 문제 3: CDN 캐시 설정

미디어 스트리밍 회사가 전 세계적으로 낮은 지연 시간으로 비디오 썸네일을 제공하려 합니다. `ap-northeast-2`에 썸네일 이미지가 담긴 S3 버킷(`media-bucket`)이 있습니다.

1. 이미지를 전 세계에 배포하기 위해 어떤 AWS 서비스를 사용하겠습니까?
2. 캐시에 어떤 TTL(Time-to-Live)을 설정하겠으며, 이유는 무엇입니까?
3. 썸네일이 업데이트될 때 캐시를 어떻게 무효화(invalidate)하겠습니까?

<details>
<summary>정답 보기</summary>

1. **Amazon CloudFront** — S3 버킷을 CloudFront 오리진(origin)으로 설정합니다. CloudFront는 전 세계 400개 이상의 엣지(edge) 위치를 가집니다. 북미, 유럽, 아시아의 사용자는 매번 서울에서 다운로드하는 대신 가장 가까운 엣지 위치에서 썸네일을 받습니다.

2. **TTL 권장: 긴 TTL (86400초 = 24시간 이상)**

   썸네일은 일반적으로 거의 변경되지 않는 정적 자산입니다. 긴 TTL은 다음을 의미합니다:
   - 엣지 위치가 오리진에 접속하지 않고 24시간 동안 캐시에서 제공합니다.
   - S3의 부하를 크게 줄입니다(오리진 요청 감소 = 비용 절감).
   - 전 세계 사용자에게 빠른 캐시 히트(cache hit)를 제공합니다.

   변경되는 썸네일(예: 사용자가 프로필 사진 업데이트 시)에는 버전이 붙은 파일명(예: `user-123-v2.jpg`)을 사용하면 이전 썸네일은 무기한 캐시에 유지되고, 새로운 URL로 새 썸네일이 로드됩니다.

3. **캐시 무효화**:
```bash
# 특정 썸네일 무효화
aws cloudfront create-invalidation \
    --distribution-id DISTRIBUTION_ID \
    --paths "/thumbnails/user-123.jpg"

# 모든 썸네일 무효화
aws cloudfront create-invalidation \
    --distribution-id DISTRIBUTION_ID \
    --paths "/thumbnails/*"
```

**비용 고려사항**: CloudFront는 무효화 경로(path) 1,000개당 $0.005를 청구합니다(월 처음 1,000개 무료). 자주 업데이트되는 자산에는 무효화 비용 방지를 위해 URL 버전 관리를 선호하세요.

</details>

### 연습 문제 4: 경로 기반 라우팅(Path-Based Routing) 규칙

마이크로서비스로 마이그레이션 중인 모놀리스를 제공하는 ALB가 있습니다. 현재 모든 트래픽이 `legacy-app` 타깃 그룹으로 이동합니다. `/api/v2/*` 요청을 새 `microservice-api` 타깃 그룹으로 라우팅하고 다른 모든 것은 `legacy-app`으로 유지해야 합니다.

ALB 리스너 규칙 설정을 설명하거나 CLI 명령어를 제공하세요.

<details>
<summary>정답 보기</summary>

```bash
# /api/v2/* → microservice-api 타깃 그룹에 규칙 추가 (우선순위 10)
# 다른 모든 트래픽은 기본 규칙(legacy-app)으로 이동
aws elbv2 create-rule \
    --listener-arn arn:aws:elasticloadbalancing:ap-northeast-2:123456789012:listener/app/my-alb/xxx/yyy \
    --priority 10 \
    --conditions '[{"Field":"path-pattern","Values":["/api/v2/*"]}]' \
    --actions '[{"Type":"forward","TargetGroupArn":"arn:aws:elasticloadbalancing:ap-northeast-2:123456789012:targetgroup/microservice-api/zzz"}]'
```

**ALB 규칙 작동 방식**:
1. 규칙은 우선순위 순서로 평가됩니다(숫자가 낮을수록 높은 우선순위).
2. 우선순위 10 규칙: 경로가 `/api/v2/*`와 일치하면 → `microservice-api`로 포워드.
3. 기본 규칙(우선순위 `default`, 항상 마지막): 다른 모든 요청 → `legacy-app`으로 포워드.

**검증**:
```bash
# 리스너의 모든 규칙 나열하여 확인
aws elbv2 describe-rules \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx
```

이 패턴(스트랭글러 피그 아키텍처, Strangler Fig Architecture)은 로드 밸런서를 트래픽 라우터로 사용하여 모놀리스를 마이크로서비스로 점진적으로 마이그레이션할 수 있게 합니다.

</details>

### 연습 문제 5: 로드 밸런서 문제 해결

개발자가 EC2 인스턴스가 실행 중이고 인스턴스의 공개 IP로 직접 접근하면 애플리케이션이 올바르게 응답하지만, ALB가 모든 요청에 대해 `502 Bad Gateway`를 반환한다고 보고합니다.

4가지 가능한 원인과 각각의 진단/수정 방법을 나열하세요.

<details>
<summary>정답 보기</summary>

1. **인스턴스가 타깃 그룹에 등록되지 않았거나 비정상(unhealthy)으로 표시됨**
   - 진단: `aws elbv2 describe-target-health --target-group-arn <ARN>`
   - 수정: 누락된 경우 인스턴스를 등록합니다. 비정상인 경우 헬스 체크 경로를 확인하고 애플리케이션이 헬스 체크 엔드포인트에서 예상 HTTP 상태 코드로 응답하는지 확인합니다.

2. **보안 그룹이 ALB에서의 트래픽을 차단함**
   - 진단: 인스턴스의 보안 그룹은 애플리케이션 포트(예: 80)에서 ALB의 보안 그룹으로부터 인바운드 트래픽을 허용해야 합니다. ALB SG 대신 `0.0.0.0/0` 또는 특정 IP만 허용하면 내부 ALB-인스턴스 트래픽이 차단됩니다.
   - 수정: 인스턴스 보안 그룹에 인바운드 규칙 추가: 소스 = ALB 보안 그룹 ID, 포트 = 애플리케이션 포트.

3. **헬스 체크 경로가 200 이외의 상태 코드 반환**
   - 진단: ALB 타깃 헬스 상태를 확인합니다. `State: unhealthy`이면 헬스 체크가 실패하고 있습니다. 엔드포인트를 수동으로 테스트: `curl http://<INSTANCE_IP>/health`.
   - 수정: 200을 반환하도록 헬스 체크 엔드포인트를 수정하거나, 실제 정상 응답 코드에 맞게 헬스 체크 설정을 업데이트합니다.

4. **애플리케이션이 잘못된 포트에서 수신 중이거나 시작되지 않음**
   - 진단: 앱이 공개 IP에서 응답하더라도 타깃 그룹에 설정된 것과 다른 포트를 사용할 수 있습니다. 인스턴스에서 확인: `ss -tlnp | grep <port>`.
   - 수정: 애플리케이션이 타깃 그룹 설정과 동일한 포트에서 수신(bind)하고 있는지 확인합니다(예: 타깃 그룹이 포트 8080을 사용하면 앱도 8080에 바인딩해야 합니다).

</details>

---

## 참고 자료

- [AWS ELB Documentation](https://docs.aws.amazon.com/elasticloadbalancing/)
- [AWS CloudFront Documentation](https://docs.aws.amazon.com/cloudfront/)
- [GCP Load Balancing](https://cloud.google.com/load-balancing/docs)
- [GCP Cloud CDN](https://cloud.google.com/cdn/docs)
