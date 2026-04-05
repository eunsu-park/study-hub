# Chaos Engineering

**이전**: [Secrets Management](./15_Secrets_Management.md) | **다음**: [Platform Engineering](./17_Platform_Engineering.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Chaos engineering 원칙을 정의하고 능동적인 장애 주입이 더 복원력 있는 시스템을 구축하는 이유 설명하기
2. 과학적 방법을 사용하여 실험 설계하기: 정상 상태 가설, 실험, 관찰, 결론
3. Chaos Monkey, Litmus 및 기타 chaos engineering 도구를 사용하여 장애 주입 구현하기
4. 실험이 실제 장애를 일으키지 않도록 영향 범위(blast radius) 제어하기
5. 제어된 조건에서 인시던트 대응을 연습하기 위한 게임 데이를 계획하고 실행하기
6. 장애 주입 유형을 분류하고 다양한 시스템 구성 요소에 적절한 실험 선택하기

---

Chaos engineering은 프로덕션에서 혼란스러운 조건을 견딜 수 있는 시스템의 능력에 대한 신뢰를 구축하기 위해 시스템을 실험하는 분야입니다. 새벽 3시에 장애가 발생하기를 기다리는 대신, chaos 엔지니어는 팀이 대응할 준비가 된 업무 시간 동안 의도적으로 장애를 주입합니다. 목표는 것을 깨뜨리는 것이 아니라 -- 약점이 당신을 찾기 전에 약점을 발견하는 것입니다.

> **비유 -- 소방 훈련**: 소방 훈련은 건물에 불을 지르지 않습니다. 비상 상황을 시뮬레이션하여 사람들이 대피 경로를 알고 있는지, 경보가 작동하는지, 출구가 차단되지 않았는지 테스트합니다. Chaos engineering은 분산 시스템을 위한 소방 훈련입니다: 장애(서버 충돌, 네트워크 분할, 디스크 가득 참)를 시뮬레이션하여 모니터링이 문제를 감지하는지, 자동화가 복구하는지, 사용자가 영향받지 않는지 확인합니다.

## 1. Chaos Engineering의 필요성

### 1.1 복원력 격차

```
What We Think Happens:          What Actually Happens:
┌──────────────┐                ┌──────────────┐
│   Service    │                │   Service    │
│   Fails      │                │   Fails      │
│      ↓       │                │      ↓       │
│  Auto-heal   │                │  Cascading   │
│  Kicks In    │                │  Failure     │
│      ↓       │                │      ↓       │
│  Users       │                │  Timeout     │
│  Unaffected  │                │  Propagation │
│              │                │      ↓       │
│              │                │  Full Outage │
│              │                │  (3 AM)      │
└──────────────┘                └──────────────┘
```

Chaos engineering이 반증하는 일반적인 가정:
- "우리의 오토 스케일링은 트래픽 급증을 처리한다" (스케일링 정책에 버그가 있으면 그렇지 않음)
- "우리의 서킷 브레이커가 cascading 장애를 방지한다" (타임아웃이 너무 높게 설정되면 그렇지 않음)
- "우리의 데이터베이스 장애 조치는 원활하다" (애플리케이션이 재연결하지 않으면 그렇지 않음)
- "우리의 모니터링은 모든 문제를 감지한다" (알림이 잘못 구성되면 그렇지 않음)

### 1.2 Chaos Engineering vs 테스팅

| 측면 | 전통적 테스팅 | Chaos Engineering |
|------|-------------|-------------------|
| **환경** | 테스트/스테이징 | 프로덕션 (선호) |
| **범위** | 알려진 장애 모드 | 알려지지 않은 장애 모드 |
| **목표** | 예상 동작 검증 | 예상치 못한 동작 발견 |
| **방법론** | 통과/실패가 있는 테스트 케이스 | 가설이 있는 실험 |
| **장애** | 시뮬레이션/모의 | 실제 (라이브 시스템에 주입) |

---

## 2. Chaos Engineering 원칙

### 2.1 다섯 가지 원칙

1. **정상 상태 동작에 대한 가설 수립**
   - 메트릭을 사용하여 "정상"이 무엇인지 정의 (오류율 < 0.1%, p99 지연시간 < 500ms)
   - 가설: "장애 X를 주입할 때, 정상 상태가 유지될 것이다"

2. **실제 세계의 이벤트를 변화시키기**
   - 실제로 발생하는 장애를 주입: 서버 충돌, 네트워크 분할, 디스크 가득 참, 높은 CPU
   - 비현실적인 시나리오를 만들지 않기

3. **프로덕션에서 실험 실행**
   - 스테이징 환경은 실제 트래픽 패턴, 데이터 볼륨, 의존성 동작이 없음
   - 프로덕션 실험이 실제 동작을 보여줌 (안전 제어와 함께)

4. **지속적으로 실행되도록 실험 자동화**
   - 일회성 실험은 한 시점만 증명; 지속적인 실험은 회귀를 감지
   - Chaos 실험을 CI/CD 파이프라인에 통합

5. **영향 범위(blast radius) 최소화**
   - 작게 시작 (인스턴스 하나, AZ 하나, canary 트래픽)
   - 실험을 즉시 중단하는 킬 스위치 보유
   - 신뢰가 커지면서 범위를 점진적으로 확장

### 2.2 실험 라이프사이클

```
┌──────────────────────────────────────────────────────────────┐
│              Chaos Experiment Lifecycle                        │
│                                                               │
│  1. Define Steady State                                       │
│     └─ "Error rate < 0.1%, p99 < 500ms, orders/min > 100"   │
│                                                               │
│  2. Hypothesize                                               │
│     └─ "If we kill 1 of 3 order-service pods, steady state   │
│         will be maintained because Kubernetes will reschedule │
│         and the load balancer will route around it"           │
│                                                               │
│  3. Design Experiment                                         │
│     └─ Kill 1 pod in production for 5 minutes                │
│     └─ Blast radius: single pod, single service              │
│     └─ Abort criteria: error rate > 1% or p99 > 2s           │
│                                                               │
│  4. Run Experiment                                            │
│     └─ Inject failure, observe metrics in real-time           │
│                                                               │
│  5. Analyze Results                                           │
│     └─ Did steady state hold? If not, why?                   │
│                                                               │
│  6. Improve                                                   │
│     └─ Fix the weakness, then re-run the experiment          │
│                                                               │
│  7. Automate                                                  │
│     └─ Add to continuous chaos suite                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 장애 주입 유형

### 3.1 분류

| 카테고리 | 장애 유형 | 예시 | 도구 |
|---------|----------|------|------|
| **인프라** | 인스턴스 종료 | VM 또는 파드 종료 | Chaos Monkey, Litmus |
| **인프라** | AZ 장애 | 전체 영역 장애 시뮬레이션 | AWS FIS, Gremlin |
| **네트워크** | 지연시간 주입 | 서비스 호출에 500ms 지연 추가 | tc, Toxiproxy, Istio |
| **네트워크** | 패킷 손실 | 네트워크 패킷의 10% 삭제 | tc, Litmus |
| **네트워크** | DNS 장애 | DNS 해석 실패 | CoreDNS 조작 |
| **네트워크** | 파티션 | 서비스 A가 서비스 B에 도달 불가 | iptables, Istio fault injection |
| **리소스** | CPU 스트레스 | 95% CPU 소비 | stress-ng, Litmus |
| **리소스** | 메모리 압박 | OOM까지 가용 메모리 소비 | stress-ng, Litmus |
| **리소스** | 디스크 가득 참 | 디스크를 100% 채움 | dd, Litmus |
| **애플리케이션** | 예외 주입 | 특정 코드 경로가 오류를 발생시키도록 강제 | Feature flag, SDK |
| **애플리케이션** | 의존성 장애 | 외부 API가 500 반환 또는 타임아웃 | Toxiproxy, Istio |
| **상태** | 시계 왜곡 | 시스템 시계를 앞/뒤로 설정 | chrony 조작 |
| **상태** | 데이터 손상 | 메시지 큐 또는 캐시의 잘못된 데이터 | 커스텀 스크립트 |

### 3.2 네트워크 장애 주입

```bash
# Using tc (traffic control) — add 200ms latency to eth0
tc qdisc add dev eth0 root netem delay 200ms 50ms distribution normal

# Add 5% packet loss
tc qdisc add dev eth0 root netem loss 5%

# Add both latency and loss
tc qdisc add dev eth0 root netem delay 200ms loss 5%

# Remove all rules
tc qdisc del dev eth0 root

# Using Toxiproxy — proxy that simulates network conditions
# Create a proxy for PostgreSQL
toxiproxy-cli create postgres-proxy -l localhost:25432 -u postgres:5432

# Add 500ms latency
toxiproxy-cli toxic add postgres-proxy -t latency -a latency=500 -a jitter=100

# Add connection timeout
toxiproxy-cli toxic add postgres-proxy -t timeout -a timeout=5000

# Simulate connection reset
toxiproxy-cli toxic add postgres-proxy -t reset_peer -a timeout=1000
```

### 3.3 Istio Fault Injection

```yaml
# Inject 500ms delay into 10% of requests to the payment service
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: payment-service
spec:
  hosts:
    - payment-service
  http:
    - fault:
        delay:
          percentage:
            value: 10.0
          fixedDelay: 500ms
      route:
        - destination:
            host: payment-service

---
# Return HTTP 500 for 5% of requests
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: payment-service
spec:
  hosts:
    - payment-service
  http:
    - fault:
        abort:
          percentage:
            value: 5.0
          httpStatus: 500
      route:
        - destination:
            host: payment-service
```

---

## 4. Chaos Monkey

### 4.1 Netflix의 Chaos Monkey

Chaos Monkey는 Netflix의 도구로 프로덕션에서 인스턴스를 무작위로 종료합니다. 첫 번째로 널리 알려진 chaos engineering 도구였으며 시스템이 개별 인스턴스 장애를 견디도록 설계되어야 한다는 원칙을 확립했습니다.

**핵심 개념:**
- 업무 시간(엔지니어가 대응 가능한 시간)에 실행
- 구성된 그룹의 무작위 인스턴스를 대상으로 함
- 팀이 처음부터 인스턴스 장애를 고려한 설계를 하도록 강제

### 4.2 Kube-Monkey (Kubernetes)

```yaml
# kube-monkey configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: kube-monkey-config
  namespace: kube-system
data:
  config.toml: |
    [kubemonkey]
    run_hour = 8              # Start killing pods at 8 AM
    start_hour = 10           # First kill window
    end_hour = 16             # Last kill window
    grace_period_sec = 5      # Grace period before SIGKILL
    cluster_dns_name = "cluster.local"
    whitelisted_namespaces = ["production", "staging"]
    blacklisted_namespaces = ["kube-system", "monitoring"]

---
# Opt-in: add labels to deployments that should be targeted
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  labels:
    kube-monkey/enabled: "enabled"
    kube-monkey/mtbf: "2"          # Mean time between failures (days)
    kube-monkey/identifier: "order-service"
    kube-monkey/kill-mode: "fixed"
    kube-monkey/kill-value: "1"    # Kill 1 pod at a time
```

---

## 5. Litmus Chaos

### 5.1 아키텍처

Litmus는 Kubernetes 네이티브 chaos engineering 프레임워크를 제공하는 CNCF 프로젝트입니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Litmus Architecture                          │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │  ChaosCenter │   │  Chaos       │   │  Chaos       │       │
│  │  (UI + API)  │──→│  Operator    │──→│  Runner      │       │
│  │              │   │  (watches    │   │  (executes   │       │
│  │              │   │   CRDs)      │   │   experiments│       │
│  └──────────────┘   └──────────────┘   └──────┬───────┘       │
│                                                │                │
│                                         ┌──────┴───────┐       │
│                                         │  Experiment  │       │
│                                         │  Pod         │       │
│                                         │  (inject     │       │
│                                         │   failure)   │       │
│                                         └──────────────┘       │
│                                                                  │
│  CRDs:                                                          │
│  - ChaosEngine (what to target)                                 │
│  - ChaosExperiment (how to inject)                              │
│  - ChaosResult (what happened)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Pod Delete 실험

```yaml
# ChaosExperiment: defines the experiment type
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosExperiment
metadata:
  name: pod-delete
  namespace: production
spec:
  definition:
    scope: Namespaced
    permissions:
      - apiGroups: [""]
        resources: ["pods"]
        verbs: ["delete", "list", "get"]
    image: litmuschaos/go-runner:latest
    args:
      - -name
      - pod-delete
    env:
      - name: TOTAL_CHAOS_DURATION
        value: "30"           # Kill pods for 30 seconds
      - name: CHAOS_INTERVAL
        value: "10"           # Every 10 seconds
      - name: FORCE
        value: "false"
      - name: PODS_AFFECTED_PERC
        value: "50"           # Kill 50% of matching pods

---
# ChaosEngine: binds experiment to a target
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: order-service-chaos
  namespace: production
spec:
  appinfo:
    appns: production
    applabel: "app=order-service"
    appkind: deployment
  chaosServiceAccount: litmus-admin
  experiments:
    - name: pod-delete
      spec:
        probe:
          # Steady-state probe: check error rate during experiment
          - name: check-error-rate
            type: promProbe
            mode: Continuous
            runProperties:
              probeTimeout: 5s
              interval: 5s
            promProbe/inputs:
              endpoint: http://prometheus:9090
              query: |
                sum(rate(http_requests_total{job="order-service",status=~"5.."}[1m]))
                / sum(rate(http_requests_total{job="order-service"}[1m])) * 100
              comparator:
                type: float
                criteria: "<="
                value: "1.0"    # Error rate must stay below 1%
```

### 5.3 네트워크 Chaos 실험

```yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: network-chaos
  namespace: production
spec:
  appinfo:
    appns: production
    applabel: "app=order-service"
    appkind: deployment
  experiments:
    - name: pod-network-latency
      spec:
        components:
          env:
            - name: NETWORK_INTERFACE
              value: "eth0"
            - name: NETWORK_LATENCY
              value: "500"          # 500ms latency
            - name: JITTER
              value: "100"          # +/- 100ms jitter
            - name: TOTAL_CHAOS_DURATION
              value: "120"          # 2 minutes
            - name: DESTINATION_IPS
              value: "10.0.1.0/24"  # Only affect traffic to DB subnet
```

---

## 6. 정상 상태 가설

### 6.1 정상 상태 정의

정상 상태 가설은 chaos 실험에서 가장 중요한 부분입니다. "정상"을 측정 가능한 용어로 정의합니다:

```yaml
# Example steady-state definition
steady_state:
  metrics:
    - name: error_rate
      query: "sum(rate(http_requests_total{status=~'5..'}[5m])) / sum(rate(http_requests_total[5m]))"
      threshold: "< 0.001"      # Less than 0.1%

    - name: p99_latency
      query: "histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))"
      threshold: "< 0.5"        # Less than 500ms

    - name: throughput
      query: "sum(rate(http_requests_total[5m]))"
      threshold: "> 100"        # More than 100 req/s

    - name: pod_restarts
      query: "sum(increase(kube_pod_container_status_restarts_total{namespace='production'}[5m]))"
      threshold: "< 1"          # No unexpected restarts

  health_endpoints:
    - url: "https://api.example.com/health"
      expected_status: 200
      timeout: 5s
```

### 6.2 중단 기준

실험을 자동으로 중단할 시점을 정의합니다:

| 메트릭 | 정상 | 경고 (계속) | 중단 (즉시 중지) |
|--------|------|------------|-----------------|
| 오류율 | < 0.1% | 0.1% - 1% | > 1% |
| p99 지연시간 | < 500ms | 500ms - 2s | > 2s |
| 파드 재시작 | 0 | 1-2 | > 2 |
| 고객 영향 | 없음 | 가시적 없음 | 사용자 대면 오류 발생 |

---

## 7. 영향 범위(Blast Radius) 제어

### 7.1 점진적 영향 범위

```
Level 1: Development/Staging
├── Single pod kill
├── No real user traffic
└── Full experiment freedom

Level 2: Production — Canary
├── Target canary deployment only
├── < 5% of traffic affected
└── Automatic abort on metric breach

Level 3: Production — Single AZ
├── Target one availability zone
├── ~33% of capacity affected
└── Monitor for cross-AZ failover

Level 4: Production — Full Scope
├── Target entire service
├── 100% of instances affected
└── Reserved for game days with full team readiness
```

### 7.2 안전 제어

| 제어 | 설명 |
|------|------|
| **킬 스위치** | 모든 chaos 실험을 즉시 중단하는 수동 버튼 |
| **시간 제한** | 최대 기간 후 실험 자동 중지 |
| **메트릭 가드** | 핵심 메트릭이 임계값을 위반하면 중단 |
| **네임스페이스 격리** | Chaos 실험이 허용된 네임스페이스만 대상 |
| **업무 시간** | 엔지니어링 팀이 가용한 시간에만 실행 (오전 9시 - 오후 4시) |
| **변경 동결** | 배포, 유지보수, 인시던트 중에는 실행하지 않음 |
| **옵트인 라벨링** | 명시적 chaos 라벨이 있는 리소스만 대상 |

---

## 8. 게임 데이

### 8.1 게임 데이란

게임 데이는 팀이 의도적으로 프로덕션(또는 프로덕션과 유사한 환경)에 장애를 주입하고 인시던트 대응을 연습하는 예정된 팀 전체의 chaos engineering 이벤트입니다.

### 8.2 게임 데이 구조

```
┌────────────────────────────────────────────────────────────────┐
│                     Game Day Schedule                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Before (1 week):                                              │
│  ├── Define scenarios and hypotheses                           │
│  ├── Notify stakeholders (leadership, support, on-call)        │
│  ├── Verify monitoring and alerting are working                │
│  ├── Prepare rollback plans                                    │
│  └── Brief the team on experiment details                      │
│                                                                │
│  During (2-4 hours):                                           │
│  ├── 09:00 — Kickoff and role assignment                       │
│  │   ├── Experiment Lead (injects failures)                    │
│  │   ├── Observers (watch metrics, take notes)                 │
│  │   └── Incident Commander (coordinates response)             │
│  ├── 09:30 — Experiment 1: Kill 1 of 3 API pods               │
│  ├── 10:00 — Experiment 2: Inject 500ms latency to DB         │
│  ├── 10:30 — Experiment 3: Simulate AZ failure                │
│  ├── 11:00 — Experiment 4: Secrets rotation under load         │
│  └── 11:30 — Restore steady state, verify recovery            │
│                                                                │
│  After (1-2 days):                                             │
│  ├── Debrief meeting (what surprised us?)                      │
│  ├── Document findings and action items                        │
│  ├── File bugs for discovered weaknesses                       │
│  └── Schedule follow-up experiments                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 8.3 게임 데이 시나리오 예시

| 시나리오 | 가설 | 관찰할 것 |
|---------|------|----------|
| **API 파드의 50% 종료** | Kubernetes가 30초 이내에 파드를 재스케줄; 오류율이 0.5% 미만 유지 | 파드 스케줄링 시간, 전환 중 요청 오류 |
| **데이터베이스 primary 장애 조치** | RDS Multi-AZ 장애 조치가 60초 이내에 완료; 앱이 자동 재연결 | 장애 조치 소요 시간, 연결 풀 복구, 데이터 일관성 |
| **Redis 캐시 클러스터 다운** | 애플리케이션이 데이터베이스 쿼리로 폴백; 지연시간 증가하지만 오류 없음 | 캐시 미스율, DB 쿼리 부하, 응답 시간 |
| **DNS 장애** | 서킷 브레이커 열림; 캐시된 응답 제공; 2분 이내에 알림 발생 | 서킷 브레이커 동작, 모니터링 감지 시간 |
| **리전 장애** | 보조 리전으로 5분 이내에 트래픽 장애 조치; 데이터 일관성 유지 | DNS 장애 조치 시간, 데이터 복제 지연, 사용자 영향 |

---

## 9. Chaos Engineering 성숙도 모델

### 9.1 성숙도 수준

```
Level 0: Ad Hoc
├── No chaos engineering
├── Failures discovered by users
└── Reactive incident response

Level 1: Initial
├── Manual experiments in staging
├── Kill individual pods/instances
├── Basic monitoring exists
└── Post-incident analysis begins

Level 2: Repeatable
├── Scheduled experiments (monthly)
├── Production experiments with safety controls
├── Steady-state hypotheses defined
├── Game days conducted quarterly
└── Findings tracked as action items

Level 3: Defined
├── Automated chaos experiments in CI/CD
├── Continuous chaos in production (Chaos Monkey)
├── Metrics-based abort criteria
├── Chaos engineering integrated into development lifecycle
└── Cross-team participation

Level 4: Optimized
├── Custom chaos experiments for business logic
├── Chaos experiments trigger automatically on every deployment
├── Full production coverage (all services, all failure modes)
├── Chaos engineering culture embedded in the organization
└── Public game day reports and knowledge sharing
```

---

## 10. 다음 단계

- [17_Platform_Engineering.md](./17_Platform_Engineering.md) - 설계부터 chaos에 복원력 있는 플랫폼 구축
- [18_SRE_Practices.md](./18_SRE_Practices.md) - 신뢰성 예산 관리를 위한 SRE 관행

---

## 연습 문제

### 연습 문제 1: 실험 설계

다음 시나리오에 대한 chaos 실험을 설계하십시오:

전자상거래 플랫폼에 inventory 서비스, payment 서비스, notification 서비스에 의존하는 checkout 서비스가 있습니다. inventory와 payment 서비스는 핵심적이며(이들 없이는 checkout 실패), notification 서비스는 비핵심적입니다(이메일 확인이 지연될 수 있음).

checkout 서비스가 notification 서비스 장애를 우아하게 처리하는지 확인하는 실험을 설계하십시오.

<details>
<summary>정답 보기</summary>

**실험: Checkout 중 Notification 서비스 장애**

**정상 상태 가설:**
- Checkout 성공률 > 99.5%
- Checkout p99 지연시간 < 2초
- 분당 주문 수 > 기준선 (현재 비율)
- Checkout 페이지에서 사용자 대면 오류 없음

**가설 진술:**
"notification 서비스가 완전히 사용 불가능할 때, checkout 작업은 성공적으로 완료됩니다. 이메일 알림은 큐에 저장되어 서비스가 복구될 때 전달됩니다. 사용자는 성공 페이지를 볼 것입니다 (오류 페이지가 아닌)."

**실험 설계:**
```yaml
# Litmus experiment: kill all notification-service pods
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: notification-outage
spec:
  appinfo:
    appns: production
    applabel: "app=notification-service"
    appkind: deployment
  experiments:
    - name: pod-delete
      spec:
        components:
          env:
            - name: TOTAL_CHAOS_DURATION
              value: "300"        # 5 minutes
            - name: PODS_AFFECTED_PERC
              value: "100"        # Kill ALL pods
            - name: CHAOS_INTERVAL
              value: "10"         # Re-kill if Kubernetes reschedules
        probe:
          - name: checkout-success-rate
            type: promProbe
            mode: Continuous
            promProbe/inputs:
              query: "sum(rate(checkout_total{status='success'}[1m])) / sum(rate(checkout_total[1m]))"
              comparator:
                type: float
                criteria: ">="
                value: "0.995"
```

**영향 범위 제어:**
- notification 서비스만 영향 (checkout, inventory, payment는 손대지 않음)
- 기간: 최대 5분
- Checkout 성공률이 99% 미만으로 떨어지면 중단 (99.5%가 아닌 -- 작은 마진 허용)
- 당직 엔지니어가 인지한 상태에서 업무 시간에 실행

**예상 결과:**
1. **시스템이 잘 설계된 경우**: Checkout 서비스가 notification 서비스에 비동기 패턴(메시지 큐)을 사용합니다. 소비자가 다운되어도 메시지가 큐에 저장됩니다. Checkout이 즉시 성공합니다. notification 서비스가 복구되면 큐의 메시지를 처리하고 이메일을 보냅니다.
2. **버그가 있는 경우**: Checkout 서비스가 notification 서비스에 동기 HTTP 호출을 합니다. 호출이 5초 후 타임아웃되고, checkout이 500 오류로 실패합니다. **이것이 수정해야 할 버그입니다**: notification 호출은 서킷 브레이커가 있는 비동기여야 합니다.

**수정 (실험 실패 시):**
- Notification 호출을 동기 HTTP에서 비동기 메시지 큐(SQS, RabbitMQ)로 변경
- 나중에 전달하기 위해 알림을 기록하는 fallback이 있는 서킷 브레이커 추가
- 수정 사항을 확인하기 위해 실험 재실행

</details>

### 연습 문제 2: 영향 범위 평가

다음 각 chaos 실험에 대해 영향 범위(낮음, 중간, 높음)를 평가하고 적절한 안전 제어 없이 무엇이 잘못될 수 있는지 식별하십시오:

1. Stateless API 서비스의 10개 파드 중 1개 종료
2. API 게이트웨이와 모든 백엔드 서비스 간에 2초 네트워크 지연시간 주입
3. Primary 데이터베이스 서버의 디스크를 100%로 채움
4. 완전한 AWS 리전 장애 시뮬레이션

<details>
<summary>정답 보기</summary>

**1. 10개 파드 중 1개 종료 -- 영향 범위: 낮음**
- 영향: 10% 용량 감소; 로드 밸런서가 죽은 파드를 우회하여 라우팅
- 제어 없는 위험: 최소. Kubernetes가 몇 초 이내에 재스케줄. 파드의 시작 시간이 길면 짧은 용량 감소.
- 잘못될 수 있는 것: 파드가 인메모리 상태(스티키 세션, 로컬 캐시)를 보유하면 해당 세션이 손실됨. PDB(Pod Disruption Budget)가 잘못 설정되면 종료가 차단될 수 있음.
- 안전: 표준 readiness probe와 PDB가 충분.

**2. 모든 백엔드에 2초 지연시간 -- 영향 범위: 높음**
- 영향: API 게이트웨이를 통한 모든 요청이 추가 2초 소요. 모든 사용자가 성능 저하를 경험.
- 제어 없는 위험: 연결 풀 고갈. API 게이트웨이가 정상 50ms 대신 2초 이상 연결을 유지하여 연결 풀을 빠르게 고갈시킴. 스레드 풀 고갈이 뒤따르며 게이트웨이가 모든 요청을 거부하기 시작. Cascading 장애: 상위 서비스(CDN, 로드 밸런서)가 타임아웃 시작.
- 잘못될 수 있는 것: 실험에 지연시간 임계값에 대한 자동 중단이 없으면 전체 플랫폼이 사용 불가능해질 수 있음.
- 안전: 모든 백엔드에 동시에 지연시간을 주입하지 마십시오. 한 번에 하나의 백엔드부터 시작하고, 100%가 아닌 비율(10%)을 사용하십시오.

**3. Primary 데이터베이스 디스크 100% 채움 -- 영향 범위: 높음 (위험)**
- 영향: 데이터베이스가 쓰기 불가. 모든 쓰기 작업 실패. 트랜잭션 롤백. WAL(Write-Ahead Log) 쓰기 불가 시 데이터 손상 가능.
- 제어 없는 위험: **데이터 손실**. 데이터베이스가 WAL을 쓸 수 없으면 백업에서 복원해야 할 수 있음. 복제가 깨질 수 있음. 복구에 수 시간이 걸릴 수 있음.
- 잘못될 수 있는 것: 가장 위험한 실험 중 하나. 파드 종료(stateless)와 달리 데이터베이스의 디스크 채움은 영구적인 데이터 손실을 초래할 수 있음.
- 안전: **프로덕션 데이터베이스에서 절대 실행하지 마십시오.** 레플리카가 있는 스테이징 데이터베이스에서 테스트. 프로덕션에서 반드시 테스트해야 한다면 데이터 디렉토리가 아닌 비핵심 볼륨(예: /tmp)을 채우십시오.

**4. AWS 리전 장애 시뮬레이션 -- 영향 범위: 매우 높음**
- 영향: 해당 리전의 모든 서비스가 사용 불가. 멀티 리전 이중화가 있는 서비스만 생존.
- 제어 없는 위험: 멀티 리전 장애 조치가 올바르게 작동하지 않으면 모든 사용자에 대한 완전한 장애. DNS 전파 지연으로 장애 조치가 올바르더라도 장애가 10분 이상 연장될 수 있음.
- 잘못될 수 있는 것: 데이터베이스 복제 지연이 0이 아니면 데이터 불일치. DNS 캐시가 장애가 발생한 리전으로 계속 라우팅할 수 있음. CDN 엣지 캐시가 stale해질 수 있음.
- 안전: 전체 팀이 준비된 게임 데이 수준의 실험으로 사전 검증된 롤백 절차와 이해관계자 통지가 필요합니다. 임의로 실행하지 마십시오.

</details>

### 연습 문제 3: 정상 상태 가설

다음 조건의 결제 처리 서비스에 대한 완전한 정상 상태 가설을 작성하십시오:
- Stripe를 통한 신용카드 결제 처리
- 99.99% 성공률 유지 필수
- p99 지연시간 SLO: 1초
- PostgreSQL에 트랜잭션 저장

메트릭, 임계값, 모니터링 쿼리를 포함하십시오.

<details>
<summary>정답 보기</summary>

```yaml
steady_state_hypothesis:
  service: payment-service
  description: >
    The payment service processes credit card charges reliably
    within latency SLOs, with all transactions persisted to
    PostgreSQL and confirmation events published.

  metrics:
    # Business metrics
    - name: payment_success_rate
      description: "Percentage of payment attempts that succeed"
      query: >
        sum(rate(payment_attempts_total{result="success"}[5m]))
        / sum(rate(payment_attempts_total[5m]))
      threshold: ">= 0.9999"
      alert_on_breach: true
      severity: critical

    - name: payment_throughput
      description: "Payments processed per minute"
      query: "sum(rate(payment_attempts_total[1m])) * 60"
      threshold: ">= 50"
      alert_on_breach: true
      severity: warning

    # Latency metrics
    - name: payment_p99_latency
      description: "99th percentile payment processing latency"
      query: >
        histogram_quantile(0.99,
          sum by (le) (rate(payment_duration_seconds_bucket[5m]))
        )
      threshold: "< 1.0"
      alert_on_breach: true
      severity: critical

    - name: payment_p50_latency
      description: "50th percentile payment processing latency"
      query: >
        histogram_quantile(0.50,
          sum by (le) (rate(payment_duration_seconds_bucket[5m]))
        )
      threshold: "< 0.3"
      alert_on_breach: false
      severity: info

    # Infrastructure metrics
    - name: db_connection_pool_utilization
      description: "PostgreSQL connection pool usage"
      query: >
        payment_db_pool_active_connections
        / payment_db_pool_max_connections
      threshold: "< 0.80"
      alert_on_breach: true
      severity: warning

    - name: db_replication_lag
      description: "PostgreSQL replication lag in seconds"
      query: "pg_replication_lag_seconds{instance='payment-db'}"
      threshold: "< 1.0"
      alert_on_breach: true
      severity: critical

    - name: stripe_api_error_rate
      description: "Stripe API call failure rate"
      query: >
        sum(rate(stripe_api_calls_total{status="error"}[5m]))
        / sum(rate(stripe_api_calls_total[5m]))
      threshold: "< 0.01"
      alert_on_breach: true
      severity: warning

    - name: pod_restart_count
      description: "Payment service pod restarts"
      query: >
        sum(increase(
          kube_pod_container_status_restarts_total{
            namespace="production",
            container="payment-service"
          }[5m]
        ))
      threshold: "== 0"
      alert_on_breach: true
      severity: critical

  health_checks:
    - name: payment_service_health
      url: "https://payment.internal/health"
      expected_status: 200
      timeout: 5s
      interval: 10s

    - name: stripe_connectivity
      url: "https://payment.internal/health/stripe"
      expected_status: 200
      timeout: 10s
      interval: 30s

  abort_criteria:
    - metric: payment_success_rate
      threshold: "< 0.999"
      action: "Immediately stop experiment and alert on-call"
    - metric: payment_p99_latency
      threshold: "> 2.0"
      action: "Immediately stop experiment"
    - metric: pod_restart_count
      threshold: "> 0"
      action: "Stop experiment, investigate crash"
```

</details>

### 연습 문제 4: 게임 데이 계획

8명의 엔지니어 팀을 위한 게임 데이를 계획하십시오. 시스템은 Kubernetes에서 실행되는 3-tier 웹 애플리케이션(프론트엔드, API, 데이터베이스)이며 AWS RDS를 사용합니다. 영향 범위가 증가하는 4개의 실험을 설계하고, 역할을 정의하며, 커뮤니케이션 계획을 수립하십시오.

<details>
<summary>정답 보기</summary>

**게임 데이: "Operation Resilience" -- Q1 2024**

**일시**: 화요일, 오전 10:00 - 오후 2:00 (업무 시간, 주중)

**역할 (8명 엔지니어):**

| 역할 | 담당자 | 책임 |
|------|--------|------|
| **게임 데이 리더** | 엔지니어 1 | 실험 조율, go/no-go 결정 |
| **실험 운영자** | 엔지니어 2 | Chaos 실험 실행, 킬 스위치 제어 |
| **메트릭 관찰자** | 엔지니어 3 | 대시보드 모니터링, 실시간 메트릭 변화 보고 |
| **로그 관찰자** | 엔지니어 4 | 로그에서 오류 감시, 실험과의 상관관계 분석 |
| **고객 영향 모니터** | 엔지니어 5 | 합성 테스트와 고객 대면 오류 페이지 모니터링 |
| **인시던트 커맨더** | 엔지니어 6 | 실험이 실제 인시던트를 유발하면 인수 |
| **기록 담당** | 엔지니어 7 | 모든 것 문서화: 타임라인, 관찰, 놀라운 점 |
| **이해관계자 연락** | 엔지니어 8 | 지원 팀 및 리더십과 소통 |

**실험 (영향 범위 증가):**

| # | 시간 | 실험 | 영향 범위 | 가설 |
|---|------|------|----------|------|
| 1 | 10:30 | 4개 API 파드 중 1개 종료 | 낮음 | K8s가 30초 이내에 재스케줄; 사용자 오류 없음 |
| 2 | 11:00 | API와 DB 사이에 500ms 지연시간 주입 | 중간 | 서킷 브레이커가 1초에 열림; 캐시된 응답 제공; p99 < 2s |
| 3 | 11:30 | 모든 Redis 캐시 파드 종료 | 중~높음 | API가 DB로 폴백; 지연시간 3배 증가하지만 오류 없음 |
| 4 | 12:00 | RDS 장애 조치 시뮬레이션 (Multi-AZ) | 높음 | 장애 조치 60초 이내 완료; 앱 재연결; 실패한 트랜잭션 5개 미만 |

**커뮤니케이션 계획:**

| 시점 | 채널 | 메시지 |
|------|------|--------|
| 1주 전 | 엔지니어링 + 지원팀 이메일 | "게임 데이가 [날짜]에 예정되어 있습니다. 고객 영향은 예상되지 않습니다. 지원 팀: 잠재적인 짧은 지연시간 증가에 유의하십시오." |
| 당일 아침 | Slack #game-day 채널 | "게임 데이가 10:00에 시작됩니다. 모든 실험은 여기에 공지됩니다. 킬 스위치 보유자: [엔지니어 2]." |
| 각 실험 전 | Slack #game-day | "실험 [N] 시작: [설명]. 영향 범위: [수준]. 중단 기준: [임계값]." |
| 각 실험 후 | Slack #game-day | "실험 [N] 완료. 결과: [통과/실패]. 핵심 관찰: [요약]." |
| 하루 마무리 | 엔지니어링 + 리더십 이메일 | "게임 데이 요약: 4개 실험, [N]개 통과, [N]개에서 문제 발견. 조치 항목: [목록]." |

**중단 프로토콜:**
1. 모든 엔지니어가 언제든 "ABORT"를 요청할 수 있음
2. 실험 운영자가 즉시 실험 중단 (킬 스위치)
3. 인시던트 커맨더가 복구 필요 여부 평가
4. 팀이 다음 실험을 계속할지 게임 데이를 종료할지 결정

**사전 준비사항:**
- [ ] 모든 모니터링 대시보드 로드 및 공유
- [ ] 스테이징에서 킬 스위치 테스트
- [ ] 스테이징에서 RDS 장애 조치 테스트
- [ ] 지원 팀 브리핑
- [ ] 각 실험에 대한 롤백 절차 문서화
- [ ] 다른 배포 또는 유지보수가 예정되지 않음

</details>

---

## 참고 자료

- [Principles of Chaos Engineering](https://principlesofchaos.org/)
- [Netflix Chaos Monkey](https://netflix.github.io/chaosmonkey/)
- [Litmus Chaos Documentation](https://litmuschaos.io/)
- [AWS Fault Injection Simulator](https://docs.aws.amazon.com/fis/)
- [Gremlin Documentation](https://www.gremlin.com/docs/)
- [Chaos Engineering Book (O'Reilly)](https://www.oreilly.com/library/view/chaos-engineering/9781492043850/)
