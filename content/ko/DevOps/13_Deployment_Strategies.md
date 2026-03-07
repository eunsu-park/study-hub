# 배포 전략

**이전**: [Distributed Tracing](./12_Distributed_Tracing.md) | **다음**: [GitOps](./14_GitOps.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 배포 전략(blue-green, canary, rolling, A/B testing)을 비교하고 위험 허용도와 인프라 제약에 따라 적절한 전략 선택하기
2. 로드 밸런서 전환을 사용하여 즉시 롤백이 가능한 blue-green 배포 구현하기
3. 자동화된 메트릭 기반 프로모션과 롤백을 갖춘 canary 릴리스 파이프라인 설계하기
4. Kubernetes에서 배포 속도와 가용성의 균형을 맞추기 위한 rolling update 파라미터 구성하기
5. A/B testing(사용자 실험)과 canary 릴리스(위험 완화)의 차이 구별하기
6. LaunchDarkly 또는 Unleash를 사용하여 배포와 릴리스를 분리하는 feature flag 구현하기

---

프로덕션에 새로운 코드를 배포하는 것은 소프트웨어 운영에서 가장 위험한 일상적 활동입니다. 모든 배포는 버그, 성능 저하 또는 장애를 도입할 위험이 있습니다. 배포 전략은 새로운 코드가 사용자에게 도달하는 방식을 제어함으로써 그 위험을 줄이기 위해 존재합니다 -- 점진적으로, 안전하게, 그리고 즉시 롤백할 수 있는 능력을 갖추어. 이 레슨에서는 주요 전략들, 그 장단점, 그리고 실용적인 구현 패턴을 다룹니다.

> **비유 -- 새 다리 테스트하기**: 도시가 오래된 다리를 교체해야 한다고 상상해 보십시오. **rolling update**는 차량이 나머지 차선을 이용하는 동안 한 번에 한 차선씩 수리하는 것입니다. **blue-green 배포**는 기존 다리 옆에 두 번째 다리를 건설하고 모든 교통을 한 번에 전환하는 것입니다. **canary 릴리스**는 먼저 새 다리로 5%의 교통만 보내고 다리가 견디는 경우에만 확대하는 것입니다. **A/B testing**은 트럭은 새 다리로, 승용차는 기존 다리로 보내어 어떤 다리가 각 유형을 더 잘 처리하는지 측정하는 것입니다.

## 1. 전략 개요

### 1.1 비교 매트릭스

| 전략 | 위험도 | 롤백 속도 | 인프라 비용 | 무중단 | 적합한 용도 |
|------|--------|----------|-----------|--------|------------|
| **Rolling Update** | 중간 | 느림 (분) | 낮음 (in-place) | 예 | Kubernetes 기본값, 상태 비저장 서비스 |
| **Blue-Green** | 낮음 | 즉시 (초) | 높음 (2배 용량) | 예 | 핵심 서비스, 데이터베이스 마이그레이션 |
| **Canary** | 매우 낮음 | 빠름 (초) | 중간 (N+few) | 예 | 고트래픽 서비스, 점진적 검증 |
| **A/B Testing** | 낮음 | 빠름 | 중간 | 예 | 기능 실험, 사용자 연구 |
| **Recreate** | 높음 | 느림 | 낮음 | **아니오** | 개발/스테이징, 레거시 모놀리스 |

### 1.2 시각적 비교

```
Rolling Update:           Blue-Green:               Canary:
v1 v1 v1 v1              v1 v1 v1 v1               v1 v1 v1 v1
v2 v1 v1 v1              v1 v1 v1 v1               v1 v1 v1 v2  (5%)
v2 v2 v1 v1              v2 v2 v2 v2  ← switch     v1 v1 v2 v2  (25%)
v2 v2 v2 v1              (instant)                  v1 v2 v2 v2  (50%)
v2 v2 v2 v2                                         v2 v2 v2 v2  (100%)
(gradual)                                           (metric-driven)
```

---

## 2. Rolling Update

### 2.1 Rolling Update 작동 원리

Rolling update는 인스턴스를 한 번에 하나(또는 소수)씩 교체합니다. 배포 과정 중 어느 시점에서든 이전 버전과 새 버전이 공존합니다.

```
Time 0:  [v1] [v1] [v1] [v1]     ← All old
Time 1:  [v2] [v1] [v1] [v1]     ← 1 new, 3 old
Time 2:  [v2] [v2] [v1] [v1]     ← 2 new, 2 old
Time 3:  [v2] [v2] [v2] [v1]     ← 3 new, 1 old
Time 4:  [v2] [v2] [v2] [v2]     ← All new
```

### 2.2 Kubernetes Rolling Update 설정

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # Max extra pods during update (25% or absolute)
      maxUnavailable: 0    # Max pods that can be unavailable (0 = always maintain full capacity)
  template:
    spec:
      containers:
        - name: webapp
          image: webapp:v2
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
      minReadySeconds: 30    # Wait 30s after ready before continuing
```

**주요 파라미터 설명:**

| 파라미터 | 효과 | 권장 |
|---------|------|------|
| `maxSurge: 1` | 롤아웃 중 최대 1개의 추가 파드 | 리소스가 제한된 클러스터에서는 낮게 유지 |
| `maxUnavailable: 0` | 원하는 레플리카 수 아래로 줄이지 않음 | 핵심 서비스에 `0` 사용 |
| `minReadySeconds: 30` | 파드가 30초간 Ready 상태여야 롤아웃 계속 | 고장난 버전의 빠른 롤아웃 방지 |
| `readinessProbe` | Ready일 때만 파드에 트래픽 전달 | rolling update에는 항상 구성 |

### 2.3 Rolling Update 명령어

```bash
# Trigger a rolling update
kubectl set image deployment/webapp webapp=webapp:v2

# Watch rollout status
kubectl rollout status deployment/webapp

# Rollback to previous version
kubectl rollout undo deployment/webapp

# Rollback to a specific revision
kubectl rollout undo deployment/webapp --to-revision=3

# View rollout history
kubectl rollout history deployment/webapp

# Pause and resume (for manual canary-like control)
kubectl rollout pause deployment/webapp
kubectl rollout resume deployment/webapp
```

### 2.4 Rolling Update의 한계

- **혼합 버전**: 롤아웃 중에는 v1과 v2가 동시에 트래픽을 처리합니다. API는 하위 호환성이 있어야 합니다.
- **느린 롤백**: 롤백도 또 다른 rolling update입니다 (v2를 v1으로 한 번에 하나씩 교체).
- **트래픽 제어 불가**: 특정 사용자나 트래픽의 일정 비율을 v2로 라우팅할 수 없습니다.

---

## 3. Blue-Green 배포

### 3.1 Blue-Green 작동 원리

두 개의 동일한 환경(blue와 green)이 존재합니다. 어느 시점에서든 하나만 프로덕션 트래픽을 처리합니다. 배포는 비활성 환경에 진행되고, 로드 밸런서 전환이 모든 트래픽을 한 번에 전환합니다.

```
Phase 1: Blue is live
┌────────────────┐      ┌────────────────┐
│  Blue (v1)     │◄─LB──│    Users       │
│  [LIVE]        │      │                │
└────────────────┘      └────────────────┘
┌────────────────┐
│  Green (idle)  │
│  [STANDBY]     │
└────────────────┘

Phase 2: Deploy v2 to Green, run smoke tests
┌────────────────┐      ┌────────────────┐
│  Blue (v1)     │◄─LB──│    Users       │
│  [LIVE]        │      │                │
└────────────────┘      └────────────────┘
┌────────────────┐
│  Green (v2)    │◄── smoke tests pass
│  [READY]       │
└────────────────┘

Phase 3: Switch traffic to Green
┌────────────────┐
│  Blue (v1)     │
│  [STANDBY]     │      ┌────────────────┐
└────────────────┘      │    Users       │
┌────────────────┐      │                │
│  Green (v2)    │◄─LB──│                │
│  [LIVE]        │      └────────────────┘
└────────────────┘

Rollback: Switch LB back to Blue (instant)
```

### 3.2 Kubernetes Service를 사용한 구현

```yaml
# Blue deployment (current production)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-blue
  labels:
    app: webapp
    version: blue
spec:
  replicas: 4
  selector:
    matchLabels:
      app: webapp
      version: blue
  template:
    metadata:
      labels:
        app: webapp
        version: blue
    spec:
      containers:
        - name: webapp
          image: webapp:v1

---
# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-green
  labels:
    app: webapp
    version: green
spec:
  replicas: 4
  selector:
    matchLabels:
      app: webapp
      version: green
  template:
    metadata:
      labels:
        app: webapp
        version: green
    spec:
      containers:
        - name: webapp
          image: webapp:v2

---
# Service points to the active deployment
apiVersion: v1
kind: Service
metadata:
  name: webapp
spec:
  selector:
    app: webapp
    version: blue    # ← Change to "green" to switch
  ports:
    - port: 80
      targetPort: 8080
```

```bash
# Switch traffic from blue to green
kubectl patch service webapp -p '{"spec":{"selector":{"version":"green"}}}'

# Rollback: switch back to blue
kubectl patch service webapp -p '{"spec":{"selector":{"version":"blue"}}}'
```

### 3.3 데이터베이스 고려사항

Blue-green 배포에서 가장 어려운 부분은 데이터베이스입니다. blue와 green 모두 동일한 데이터베이스 스키마와 호환되어야 합니다.

**안전한 마이그레이션 패턴:**
1. **Expand**: 새 컬럼/테이블 추가 (하위 호환) -- v1과 v2 모두 작동
2. **Migrate**: v2 (green) 배포 및 트래픽 전환
3. **Contract**: v1 (blue) 폐기 후 이전 컬럼/테이블 제거

```
Unsafe: ALTER TABLE users RENAME COLUMN name TO full_name;
        → v1 breaks immediately because it references "name"

Safe (3-step):
Step 1: ALTER TABLE users ADD COLUMN full_name VARCHAR;
        UPDATE users SET full_name = name;
        → Both v1 (reads "name") and v2 (reads "full_name") work

Step 2: Switch traffic to v2

Step 3: ALTER TABLE users DROP COLUMN name;
        → Only after v1 is fully decommissioned
```

---

## 4. Canary 릴리스

### 4.1 Canary 릴리스 작동 원리

Canary 릴리스는 프로덕션 트래픽의 일부를 새 버전으로 라우팅합니다. 메트릭을 모니터링하고, canary가 정상이면 비율을 점진적으로 증가시킵니다. 메트릭이 악화되면 canary를 롤백합니다.

```
Phase 1: 5% canary
┌────────────────────────────────┐
│  v1 (95% traffic)              │
│  [████████████████████       ] │
│  v2 (5% traffic - canary)      │
│  [█                          ] │
└────────────────────────────────┘
    ↓ Metrics OK? (error rate, latency, saturation)

Phase 2: 25% canary
┌────────────────────────────────┐
│  v1 (75% traffic)              │
│  [███████████████            ] │
│  v2 (25% traffic)              │
│  [█████                      ] │
└────────────────────────────────┘
    ↓ Metrics OK?

Phase 3: 50% canary
... continue until 100%
```

### 4.2 Istio Service Mesh를 사용한 Canary

```yaml
# Istio VirtualService for canary traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: webapp
spec:
  hosts:
    - webapp
  http:
    - route:
        - destination:
            host: webapp
            subset: stable
          weight: 95
        - destination:
            host: webapp
            subset: canary
          weight: 5

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: webapp
spec:
  host: webapp
  subsets:
    - name: stable
      labels:
        version: v1
    - name: canary
      labels:
        version: v2
```

### 4.3 Flagger를 사용한 자동화된 Canary

Flagger는 Kubernetes에서 canary 분석과 프로모션을 자동화합니다:

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: webapp
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: webapp
  service:
    port: 80
    targetPort: 8080
  analysis:
    # Canary analysis schedule
    interval: 1m            # Check metrics every minute
    threshold: 5            # Max failed checks before rollback
    maxWeight: 50           # Max traffic percentage to canary
    stepWeight: 10          # Increase by 10% each step

    # Success criteria
    metrics:
      - name: request-success-rate
        thresholdRange:
          min: 99            # At least 99% success rate
        interval: 1m
      - name: request-duration
        thresholdRange:
          max: 500           # p99 latency < 500ms
        interval: 1m

    # Webhooks for notifications
    webhooks:
      - name: slack-notification
        type: event
        url: http://slack-notifier/
```

**Flagger 진행 과정:**
```
Step 1:  canary weight  0% → 10%   (check metrics for 1 minute)
Step 2:  canary weight 10% → 20%   (check metrics for 1 minute)
Step 3:  canary weight 20% → 30%   (check metrics for 1 minute)
Step 4:  canary weight 30% → 40%   (check metrics for 1 minute)
Step 5:  canary weight 40% → 50%   (check metrics for 1 minute)
Step 6:  promote canary → stable (scale down old version)

If any step fails metrics check 5 times → automatic rollback
```

---

## 5. A/B Testing

### 5.1 A/B Testing vs Canary

| 측면 | Canary 릴리스 | A/B Testing |
|------|-------------|-------------|
| **목표** | 배포 위험 감소 | 사용자 행동에 대한 기능 영향 측정 |
| **라우팅** | 트래픽의 무작위 비율 | 특정 사용자 세그먼트 (코호트) |
| **메트릭** | 오류율, 지연시간, 포화도 | 전환율, 매출, 참여도 |
| **기간** | 분에서 시간 단위 | 일에서 주 단위 |
| **결정** | 자동화 (메트릭 임계값) | 데이터 과학 분석 (통계적 유의성) |

### 5.2 헤더 기반 라우팅을 사용한 A/B Testing

```yaml
# Istio VirtualService: route based on user cohort header
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: webapp
spec:
  hosts:
    - webapp
  http:
    # Users in experiment group B see new checkout flow
    - match:
        - headers:
            x-experiment-group:
              exact: "checkout-v2"
      route:
        - destination:
            host: webapp
            subset: experiment-b
    # Everyone else sees the current version
    - route:
        - destination:
            host: webapp
            subset: control
```

### 5.3 A/B Testing 아키텍처

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  User    │────→│  API Gateway │────→│ Experiment   │
│  Request │     │  / CDN       │     │ Assignment   │
└──────────┘     └──────────────┘     │ Service      │
                                       └──────┬───────┘
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                       ┌──────────┐    ┌──────────┐    ┌──────────┐
                       │ Control  │    │ Variant A│    │ Variant B│
                       │ (v1)     │    │ (v2a)    │    │ (v2b)    │
                       └──────────┘    └──────────┘    └──────────┘
                              │               │               │
                              └───────────────┼───────────────┘
                                              ▼
                                       ┌──────────┐
                                       │Analytics │
                                       │ Pipeline │
                                       │(measure  │
                                       │ impact)  │
                                       └──────────┘
```

---

## 6. Feature Flag

### 6.1 Feature Flag란

Feature flag(feature toggle)는 배포와 릴리스를 분리합니다. 새로운 기능의 코드가 프로덕션에 배포되지만 플래그 뒤에 숨겨집니다. 플래그가 재배포 없이 누가 기능을 볼 수 있는지를 제어합니다.

```
Traditional:  Code Change → Deploy → Release (all at once)

Feature Flag: Code Change → Deploy (flag OFF) → Release to 5% → Release to 50% → Release to 100%
                                                      ↑                  ↑              ↑
                                                  Flag toggle        Flag toggle    Remove flag
```

### 6.2 Feature Flag 유형

| 유형 | 수명 | 목적 | 예시 |
|------|------|------|------|
| **Release toggle** | 일~주 | 새 기능의 점진적 출시 | 새로운 결제 흐름 |
| **Experiment toggle** | 주~월 | A/B testing | 버튼 색상 실험 |
| **Ops toggle** | 영구 | 성능 저하 모드용 킬 스위치 | 부하 시 추천 비활성화 |
| **Permission toggle** | 영구 | 사용자 역할/플랜별 기능 접근 | 유료 사용자 전용 프리미엄 기능 |

### 6.3 LaunchDarkly

```python
# pip install launchdarkly-server-sdk
import ldclient
from ldclient.config import Config

# Initialize the client
ldclient.set_config(Config("sdk-key-xxx"))
client = ldclient.get()

# Evaluate a feature flag
def get_checkout_page(user):
    # Create user context
    context = {
        "key": user.id,
        "email": user.email,
        "custom": {
            "plan": user.plan,       # "free", "pro", "enterprise"
            "country": user.country,
        }
    }

    # Check flag value
    show_new_checkout = client.variation("new-checkout-flow", context, False)

    if show_new_checkout:
        return render_template("checkout_v2.html")
    else:
        return render_template("checkout_v1.html")
```

### 6.4 Unleash (오픈소스 대안)

```python
# pip install UnleashClient
from UnleashClient import UnleashClient

client = UnleashClient(
    url="http://unleash:4242/api",
    app_name="webapp",
    custom_headers={"Authorization": "default:development.unleash-insecure-api-token"},
)
client.initialize_client()

def get_checkout_page(user):
    context = {
        "userId": user.id,
        "properties": {
            "plan": user.plan,
            "country": user.country,
        }
    }

    if client.is_enabled("new-checkout-flow", context):
        return render_template("checkout_v2.html")
    else:
        return render_template("checkout_v1.html")
```

### 6.5 Feature Flag 모범 사례

| 사례 | 설명 |
|------|------|
| **단기 플래그** | 전체 출시 후 2주 이내에 release toggle 제거 |
| **플래그 명명** | 설명적인 이름 사용: `enable-new-checkout-flow`, `flag-123` 아님 |
| **기본값 off** | 새 플래그는 기본적으로 `false` (안전 기본값) |
| **플래그 문서화** | 소유자, 생성 날짜, 예상 제거 날짜 추적 |
| **양쪽 경로 테스트** | 단위 테스트에서 플래그 on과 off 모두 커버 |
| **모니터링** | 플래그 평가가 실패할 때 알림 (SDK 장애) |
| **정리** | 플래그 제거를 기술 부채로 일정 관리; 오래된 플래그는 복잡성을 추가 |

---

## 7. Dark Launch

### 7.1 Dark Launch란

Dark launch는 새로운 기능을 프로덕션에 배포하고 실제 프로덕션 트래픽을 해당 기능으로 라우팅하지만, 결과는 폐기됩니다 (사용자에게 표시되지 않음). 이를 통해 사용자에게 영향을 주지 않으면서 실제 부하 조건에서 새 코드를 테스트합니다.

```
┌──────────────────────────────────────────────────────────┐
│                     Dark Launch                           │
│                                                          │
│  User Request ──→ ┌──────────┐                          │
│                   │ Current  │──→ Response to User       │
│                   │ Service  │                           │
│                   │ (v1)     │                           │
│                   └──────────┘                           │
│                        │                                 │
│                   (mirror traffic)                        │
│                        ▼                                 │
│                   ┌──────────┐                           │
│                   │ Shadow   │──→ Response discarded     │
│                   │ Service  │    (logged for analysis)  │
│                   │ (v2)     │                           │
│                   └──────────┘                           │
└──────────────────────────────────────────────────────────┘
```

### 7.2 Istio를 사용한 트래픽 미러링

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: webapp
spec:
  hosts:
    - webapp
  http:
    - route:
        - destination:
            host: webapp
            subset: stable
      mirror:
        host: webapp
        subset: shadow
      mirrorPercentage:
        value: 100.0      # Mirror 100% of traffic
```

### 7.3 Dark Launch 사용 사례

| 사용 사례 | 설명 |
|----------|------|
| **성능 테스트** | 사용자에게 영향을 주지 않으면서 v2가 프로덕션 부하를 처리하는지 검증 |
| **데이터 검증** | v2 응답을 v1 응답과 비교 (diff 분석) |
| **데이터베이스 마이그레이션** | 새 데이터베이스에 shadow-write하고 결과 비교 |
| **ML 모델 검증** | 프로덕션 데이터에 새 모델을 실행하고 예측 비교 |

---

## 8. 전략 선택 가이드

### 8.1 결정 프레임워크

```
                        Start
                          │
                    ┌─────┴─────┐
                    │ Need zero │
                    │ downtime? │
                    └─────┬─────┘
                     No   │   Yes
                     │    │
              ┌──────┘    └──────────────────────┐
              ▼                                   │
         Recreate                          ┌──────┴──────┐
         (simplest)                        │ High-traffic│
                                           │ service?    │
                                           └──────┬──────┘
                                            No    │   Yes
                                            │     │
                                     ┌──────┘     └──────────────┐
                                     ▼                            │
                              Rolling Update              ┌──────┴──────┐
                              (Kubernetes                 │ Need instant│
                               default)                   │ rollback?   │
                                                          └──────┬──────┘
                                                           No    │   Yes
                                                           │     │
                                                    ┌──────┘     └──────┐
                                                    ▼                    ▼
                                              Canary               Blue-Green
                                              (gradual,            (instant
                                               metric-driven)      switch)
```

### 8.2 서비스 유형별 전략

| 서비스 유형 | 권장 전략 | 이유 |
|------------|----------|------|
| **Stateless API** | Canary 또는 Rolling Update | 확장이 용이하고, 세션 상태 없음 |
| **Stateful 서비스** | Blue-Green | 데이터베이스 호환성에 신중한 전환 필요 |
| **프론트엔드/SPA** | Blue-Green 또는 Feature Flag | 사용자가 혼합된 UI 버전을 보지 않아야 함 |
| **백그라운드 워커** | Rolling Update | 사용자 대면 트래픽 없음, 낮은 위험도 |
| **데이터 파이프라인** | Blue-Green | 전환 전 출력 검증 필요 |
| **모바일 앱** | Feature Flag | 사용자 기기를 강제 업데이트할 수 없음 |

---

## 9. 다음 단계

- [14_GitOps.md](./14_GitOps.md) - ArgoCD와 Flux를 사용한 선언적 배포
- [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) - 배포 복원력 검증

---

## 연습 문제

### 연습 문제 1: 전략 선택

각 시나리오에 대해 배포 전략을 추천하고 선택 이유를 설명하십시오:

1. 전자상거래 회사가 새로운 결제 게이트웨이 통합을 배포합니다. 버그는 결제 실패와 매출 손실로 이어질 수 있습니다.
2. 소셜 미디어 플랫폼이 새로운 알고리즘 피드가 시간순 피드 대비 사용자 참여를 증가시키는지 테스트하려 합니다.
3. 제한된 인프라 예산을 가진 스타트업이 stateless REST API에 업데이트를 배포해야 합니다.
4. 은행이 핵심 뱅킹 애플리케이션을 Oracle에서 PostgreSQL로 마이그레이션해야 합니다.

<details>
<summary>정답 보기</summary>

1. **Canary 릴리스** -- 결제 처리는 고위험입니다. Canary는 성공률, 지연시간, 오류율을 모니터링하면서 결제 트래픽의 1-5%를 새 게이트웨이로 라우팅합니다. canary가 더 높은 실패율을 보이면 대부분의 사용자가 영향받기 전에 자동으로 롤백됩니다. Blue-green도 가능하지만, canary가 더 점진적인 위험 감소를 제공합니다.

2. **Feature flag를 사용한 A/B testing** -- 이것은 배포 위험 질문이 아니라 사용자 행동 실험입니다. Feature flag를 사용하여 사용자를 대조군(시간순)과 변형군(알고리즘) 코호트에 할당합니다. 통계적 유의성에 도달하기 위해 2-4주간 실험을 실행합니다. 참여 메트릭(사이트 체류 시간, 읽은 게시물, 재방문)을 측정합니다. Canary 릴리스는 영구적인 사용자 코호트가 아닌 무작위 트래픽을 라우팅하므로 여기서는 작동하지 않습니다.

3. **Rolling update** -- 제한된 인프라에서의 stateless REST API의 경우, rolling update가 가장 적합합니다. 추가 인프라가 필요 없고(blue-green과 같은 두 번째 환경 불필요), Kubernetes에 기본으로 내장되어 있으며, 무중단 배포를 제공합니다. 단점은 느린 롤백이지만, 트래픽이 적은 스타트업에는 수용 가능합니다.

4. **Dark launch를 사용한 blue-green 배포** -- 먼저 dark launch를 실행합니다: 모든 프로덕션 트래픽을 PostgreSQL 환경에 미러링하고 쿼리 결과를 Oracle과 비교합니다. 동일성이 확인되면, blue-green을 사용하여 전환합니다. Blue-green은 전환 후 문제가 발견되면 Oracle로의 즉시 롤백을 허용합니다. 이것은 데이터베이스 마이그레이션에서 즉시 롤백을 제공하는 유일한 전략입니다 -- canary는 두 데이터베이스로 인한 split-brain 문제를 초래할 수 있습니다.

</details>

### 연습 문제 2: Kubernetes Rolling Update

배포에 4개의 레플리카가 있고 `maxSurge: 1`과 `maxUnavailable: 1`로 구성되어 있습니다. v1에서 v2로 업데이트할 때의 단계별 롤아웃 프로세스를 설명하십시오. 각 단계에서 존재하는 파드 수와 트래픽을 처리하는 파드 수를 포함하십시오.

<details>
<summary>정답 보기</summary>

`replicas: 4`, `maxSurge: 1`, `maxUnavailable: 1`에서:
- 최대 파드 수: 4 + 1 = 5
- 최소 가용 파드 수: 4 - 1 = 3

**단계별:**

| 단계 | v1 파드 | v2 파드 | 합계 | 가용 | 작업 |
|------|---------|---------|------|------|------|
| 0 | 4 running | 0 | 4 | 4 | 초기 상태 |
| 1 | 3 running, 1 terminating | 1 creating | 5 | 3 | v2 스케일 업, v1 스케일 다운 동시 |
| 2 | 3 running | 1 ready | 4 | 4 | v2 파드 readiness probe 통과 |
| 3 | 2 running, 1 terminating | 1 ready, 1 creating | 4 | 3 | 다음 v1 파드 종료, 다음 v2 파드 시작 |
| 4 | 2 running | 2 ready | 4 | 4 | 두 번째 v2 파드 readiness probe 통과 |
| 5 | 1 running, 1 terminating | 2 ready, 1 creating | 4 | 3 | 패턴 계속 |
| 6 | 1 running | 3 ready | 4 | 4 | 세 번째 v2 파드 ready |
| 7 | 0 (terminating) | 3 ready, 1 creating | 4 | 3 | 마지막 v1 파드 종료 |
| 8 | 0 | 4 ready | 4 | 4 | 롤아웃 완료 |

**핵심 관찰 사항:**
- 시스템은 항상 최소 3개의 가용 파드를 유지합니다 (`maxUnavailable: 1` 충족).
- 시스템은 총 5개 파드를 초과하지 않습니다 (`maxSurge: 1` 충족).
- `maxSurge`와 `maxUnavailable`이 모두 0이 아닌 값이므로 새 파드 생성과 이전 파드 종료를 동시에 수행하여 롤아웃이 더 빠르게 진행됩니다.

</details>

### 연습 문제 3: Canary 메트릭

API 서비스에 대한 canary를 배포합니다. canary를 평가하는 데 사용할 메트릭, 자동 프로모션 임계값, 자동 롤백 임계값을 정의하십시오.

<details>
<summary>정답 보기</summary>

**Canary 평가 메트릭:**

| 메트릭 | 프로모션 임계값 | 롤백 임계값 | 소스 |
|--------|--------------|------------|------|
| **오류율** | < 0.5% (stable과 동일) | > 1% 또는 > 2x stable | Prometheus: `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])` |
| **p99 지연시간** | < 500ms | > 1000ms 또는 > 2x stable | Prometheus: `histogram_quantile(0.99, ...)` |
| **p50 지연시간** | < 100ms | > 200ms 또는 > 2x stable | Prometheus: `histogram_quantile(0.50, ...)` |
| **CPU 사용률** | < 80% | > 95% | Prometheus: container CPU metrics |
| **메모리 사용률** | < 80% | > 90% | Prometheus: container memory metrics |
| **패닉/재시작 횟수** | 0 | > 0 | Kubernetes: `kube_pod_container_status_restarts_total` |

**평가 방법:**
1. canary 메트릭을 stable 버전과 비교합니다 (절대 임계값만이 아닌).
2. 상대 비교 사용: "canary 오류율 < 1.5x stable 오류율"은 절대 비율이 낮더라도 회귀를 감지합니다.
3. 프로모션을 위해서는 모든 메트릭이 통과해야 합니다. 단일 메트릭 실패가 롤백을 트리거합니다.
4. 통계적으로 의미 있는 데이터를 수집하기 위해 각 가중치 단계에서 최소 5분을 기다립니다.

**Flagger 메트릭 템플릿:**
```yaml
metrics:
  - name: error-rate
    templateRef:
      name: error-rate
    thresholdRange:
      max: 1     # Max 1% error rate
    interval: 1m

  - name: latency-p99
    templateRef:
      name: latency-p99
    thresholdRange:
      max: 500   # Max 500ms p99
    interval: 1m
```

</details>

### 연습 문제 4: Feature Flag 마이그레이션

팀에 프로덕션에 47개의 feature flag가 누적되어 있으며, 일부는 1년 이상 된 것입니다. 오래된 플래그를 정리하고 향후 누적을 방지하는 계획을 설계하십시오.

<details>
<summary>정답 보기</summary>

**즉각적인 정리 계획:**

1. **47개 플래그 전수 감사**: 각각을 다음과 같이 분류:
   - **완전 출시됨** (플래그가 2주 이상 100% 사용자에게 ON): 플래그와 이전 코드 경로 제거.
   - **방치됨** (플래그가 3개월 이상 변경되지 않음): 소유자에게 연락. 1주 내 응답이 없으면 플래그 제거 (현재 기본 동작 유지).
   - **활성 실험**: 유지하되 검토 날짜 설정.
   - **Ops toggle**: 유지 (설계상 영구), 목적 문서화.

2. **제거 우선순위**: `100% ON` 상태인 플래그부터 시작 -- 새 코드 경로가 이미 검증되었으므로 제거 위험이 없습니다.

3. **코드 변경**: 제거되는 각 플래그에 대해 삭제:
   - 플래그 평가 코드
   - 이전 코드 경로 (데드 코드)
   - 이전 경로에 대한 관련 테스트
   - 플래그 관리 플랫폼에서의 플래그 정의

**방지 정책:**

| 규칙 | 설명 |
|------|------|
| **만료 날짜** | 모든 release toggle에 제거 날짜 필수 (전체 출시 후 최대 30일) |
| **소유자 지정** | 모든 플래그에 정리를 담당하는 소유자 지정 |
| **자동 알림** | 플래그가 14일 이상 100% ON 상태일 때 소유자에게 알림 |
| **CI 검사** | 총 플래그 수가 임계값(예: 20)을 초과하면 빌드 실패 |
| **검토 프로세스** | 오래된 플래그를 식별하기 위한 월간 플래그 검토 회의 (15분) |
| **명명 규칙** | 플래그 이름에 생성 날짜 포함: `2024-03-checkout-v2` |

**추적 메트릭**: "플래그 부채" = 100% on 상태인 30일 이상 된 플래그 수. 목표: 0.

</details>

---

## 참고 자료

- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Flagger - Progressive Delivery](https://flagger.app/)
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [LaunchDarkly Documentation](https://docs.launchdarkly.com/)
- [Unleash Documentation](https://docs.getunleash.io/)
- [Martin Fowler - Feature Toggles](https://martinfowler.com/articles/feature-toggles.html)
