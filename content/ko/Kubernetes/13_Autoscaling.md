# 13. 오토스케일링(Autoscaling)

**이전**: [어드미션 컨트롤러](./12_Admission_Controllers.md) | **다음**: [관측 가능성](./14_Observability.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 내장, 커스텀, 외부 메트릭을 사용하여 Horizontal Pod Autoscaler(HPA) v2를 구성할 수 있다
2. 워크로드 적정 규모 조정을 위해 Vertical Pod Autoscaler(VPA)를 배포하고 튜닝할 수 있다
3. Cluster Autoscaler를 설정하고 그 의사 결정 프로세스를 이해할 수 있다
4. KEDA를 사용하여 이벤트 소스(큐, 스트림, 데이터베이스)를 기반으로 워크로드를 스케일링할 수 있다
5. 프로덕션 환경을 위한 비용 인식 및 예측 스케일링 전략을 구현할 수 있다

---

Kubernetes의 핵심 약속 중 하나는 탄력적 스케일링(elastic scaling) -- 수요에 따라 컴퓨팅 리소스를 자동으로 조정하는 능력입니다. 그러나 Kubernetes의 오토스케일링은 단일 기능이 아니라 서로 다른 수준에서 작동하는 세 가지 구별되는 구성 요소를 가진 계층형 시스템입니다. Horizontal Pod Autoscaling은 Pod 레플리카 수를 조정하고, Vertical Pod Autoscaling은 컨테이너별 리소스 요청과 제한을 조정하며, Cluster Autoscaling은 노드 수를 조정합니다. 이 계층들이 원활하게 함께 작동하도록 하는 것은 비용 효율성과 안정성 모두에 필수적입니다.

YAML에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 쿠버네티스가 스케일링을 세 독립 계층(HPA, VPA, Cluster Autoscaler)으로 분리한 이유, HPA가 15초마다 사용하는 폐루프 제어 공식, 메트릭 지연이 실세계 스케일링 lag을 지배하는 이유, 그리고 KEDA가 비-CPU 워크로드에 가져오는 이벤트 주도 확장을 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. Horizontal Pod Autoscaler (HPA) v2](#1-horizontal-pod-autoscaler-hpa-v2)
- [2. 커스텀 메트릭과 외부 메트릭](#2-custom-metrics-and-external-metrics)
- [3. Vertical Pod Autoscaler (VPA)](#3-vertical-pod-autoscaler-vpa)
- [4. Cluster Autoscaler](#4-cluster-autoscaler)
- [5. KEDA (Kubernetes Event-Driven Autoscaling)](#5-keda-kubernetes-event-driven-autoscaling)
- [6. Prometheus 메트릭을 이용한 스케일링](#6-scaling-with-prometheus-metrics)
- [7. 예측 오토스케일링](#7-predictive-autoscaling)
- [8. 비용 인식 스케일링](#8-cost-aware-scaling)
- [9. 스케일링 모범 사례](#9-scaling-best-practices)
- [연습문제](#exercises)

---

## 이론과 원리

쿠버네티스 오토스케일링은 단일 기능이 아닙니다 — 각각 자신의 측정·결정·액추에이션을 가진 **다른 계층에서 작동하는 세 개의 독립 제어 루프**입니다:

- **HPA**(Horizontal Pod Autoscaler)는 파드별 메트릭에 기반하여 "이 Deployment의 레플리카가 몇 개여야 하는가"를 결정합니다.
- **VPA**(Vertical Pod Autoscaler)는 과거 사용량에 기반하여 "각 파드가 얼마의 CPU/메모리를 요청해야 하는가"를 결정합니다.
- **Cluster Autoscaler**는 pending 파드가 스케줄될 수 있는지에 기반하여 "클러스터에 몇 개의 노드가 있어야 하는가"를 결정합니다.

이 계층들은 신중히 합성되어야 합니다 — HPA가 스케일 아웃, 더 많은 파드가 Pending이 됨, Cluster Autoscaler가 노드 추가, 새 파드 스케줄. 순서나 메트릭이 잘못되면 oscillation, 용량 손실, 또는 폭주 비용을 얻습니다. 이 섹션은 HPA 제어 공식, 반응성을 제한하는 메트릭 파이프라인 지연, 적정 사이징에서의 VPA 역할, Cluster Autoscaler의 노드 그룹 추상, 그리고 KEDA가 큐 바운드 워크로드에 추가하는 이벤트 주도 모델을 설명합니다.

### A. HPA — 특정 공식을 가진 폐루프 컨트롤러

HPA는 15초마다(기본 `--horizontal-pod-autoscaler-sync-period`) 조정합니다. 각 사이클:

1. Metrics API(기본: CPU에 `metrics.k8s.io`, 앱 메트릭에 `custom.metrics.k8s.io`, 큐 길이 등에 `external.metrics.k8s.io`)를 통해 대상 Deployment를 받치는 모든 파드의 현재 메트릭 값 읽기.
2. 각 메트릭 소스에 대해 계산:
   ```
   desiredReplicas = ceil(currentReplicas × currentMetric / targetMetric)
   ```
   예 — CPU 50%를 타겟, 4개 파드 평균 80%이면 `ceil(4 × 80 / 50) = 7`.
3. 여러 메트릭이 구성되면, 메트릭별 desired replicas의 **최댓값** 취함(어느 메트릭이든 타겟 초과 → 스케일 업).
4. tolerance band 적용 — 변경이 현재의 ±10% 이내이면 아무것도 하지 않음(flapping 회피).
5. behavior 정책 적용 — 분당 최대 스케일 업 비율, 안정화 윈도우 등.
6. 새 레플리카 수를 Deployment에 씀.

두 결과:

- **스케일 업은 적극적, 스케일 다운은 보수적.** 기본 스케일 다운 안정화 윈도우는 5분 — HPA는 파드를 제거하기 전에 일시적 dip이 아닌지 확인하기 위해 기다립니다. 스케일 업은 0초 기본값 — 부하에 즉시 반응.
- **공식은 P-컨트롤러(비례만).** 자체적으로는 진동 워크로드를 예측하거나 평활할 수 없습니다. 날카로운 일일 패턴이 있는 워크로드는 변화율을 제한하기 위한 `behavior` 구성(HPA v2에서 도입)의 혜택을 봅니다.

흔한 함정 — 병목이 다른 것(DB 연결, 큐 깊이)일 때 CPU로 스케일하면, HPA가 실제로 CPU 바운드가 아닌 파드를 스케일하여 용량을 낭비합니다. 커스텀 메트릭(§A 이어서)이나 KEDA(§D)가 이를 해결합니다.

### B. 메트릭 파이프라인 — 반응성을 제한하는 지연

HPA는 watch하는 메트릭만큼만 빠릅니다. 기본 파이프라인:

```
파드 cgroup → kubelet (10초마다, 기본 --housekeeping-interval)
          → metrics-server (기본 60초마다 kubelet에서 스크랩)
          → Metrics API (HPA가 aggregated API로 읽음)
```

따라서 HPA의 15초 조정에도 불구하고, HPA가 보는 가장 신선한 CPU 측정은 **60+초 오래된** 것일 수 있습니다. 기본 구성에서 종단 간 "부하 도착 → 파드 스케일 업" lag은 보통 60–120초이지 15초가 아닙니다.

더 빡빡한 루프를 위해서는 metrics-server 스크랩 간격을 줄이지만(`--metric-resolution=15s`) API 서버 부하 비용을 치릅니다. 대안적으로, 커스텀 메트릭 어댑터(Prometheus Adapter, Datadog)는 모니터링 시스템에서 직접 읽으며, 그것은 이미 더 짧은 스크랩 간격을 가질 수 있습니다.

외부 소스(큐 길이, DB 연결 수)에 대해서는 Custom Metrics API 또는 External Metrics API가 HPA에 노출합니다. Prometheus Adapter가 가장 흔한 브릿지입니다 — Prometheus에 메트릭을 쓰고, 어댑터를 `custom.metrics.k8s.io/v1beta1/<resource>/<metric>`으로 노출하도록 구성하고, HPA에서 참조. 이것이 파드당 초당 요청, 큐 깊이, P99 지연, 또는 어떤 비즈니스 KPI로든 스케일하는 방법입니다.

### C. VPA와 Cluster Autoscaler — 다른 두 계층

**VPA**는 과거 리소스 사용량을 관찰하고 컨테이너당 `requests`와 `limits`를 권장합니다. 세 모드:

- `Off` — 권장만 계산(`vpa.status.recommendation`으로 표시); 사람이 읽음.
- `Initial` — 파드 생성 시에만 권장 적용; 기존 파드는 설정 유지.
- `Auto` — 새 권장을 적용하기 위해 파드 축출(PDB 존중). 파괴적이지만 완전 자동.

같은 메트릭(CPU)을 사용하는 같은 워크로드의 VPA와 HPA는 알려진 함정입니다 — 서로 싸웁니다. 메모리에 VPA + CPU에 HPA를 사용하거나, HPA가 수평 스케일하는 동안 수동 사이징을 알리기 위해 `Off` 모드의 VPA를 사용하세요.

VPA의 다른 역할은 수평 스케일할 수 없는 **배치 및 상태 워크로드 적정 사이징**입니다. 때로는 8GB, 야간 배치 동안 32GB가 필요한 데이터베이스는 완벽한 VPA 후보입니다(prod에서는 Off + 수동 모드, 또는 재시작이 허용되면 Initial 모드).

**Cluster Autoscaler (CA)**는 unschedulable 파드를 watch합니다. 스케줄러가 어떤 노드에도 자리가 없어 `Pending`을 보고하면, CA는 시뮬레이트합니다 — "그룹 X의 노드를 추가하면, 이 파드가 맞을까?" 그러면 CA는 클라우드 프로바이더에 노드 프로비저닝을 요청합니다(클라우드의 auto-scaling group / managed node group / VM scale set을 통해). 노드가 `--scale-down-unneeded-time`(기본 10분) 동안 낮은 사용률을 가지고 그 파드들이 다른 곳에 맞을 수 있으면, CA는 cordon하고 drain한 다음 클라우드에 제거를 요청합니다.

CA는 CPU나 메모리를 직접 보지 *않습니다* — *requests vs allocatable*을 봅니다. 따라서 파드의 CPU request가 낮지만 실제 사용이 높으면, CA는 노드를 추가하지 않습니다 — 그러나 파드는 과할당된 노드에 고정되어 throttle됩니다. **request의 적정 사이징이 CA를 동작하게 만듭니다.**

### D. KEDA — HPA가 놓치는 케이스를 위한 이벤트 주도 스케일링

HPA는 파드가 *노출하는* 메트릭(CPU, 메모리, 커스텀)에 기반하여 스케일합니다. 그러나 많은 워크로드는 파드가 모르는 **외부 이벤트**에 기반하여 스케일되어야 합니다:

- 큐에 10,000개 메시지 → consumer를 띄움.
- Kafka 토픽에 consumer lag → 더 많은 파티션 consumer 추가.
- 02:00에 100개 파드가 필요한 스케줄된 배치 작업 → 작업 도착 전에 스케일.

KEDA(Kubernetes Event-Driven Autoscaling)는 이러한 외부 소스를 HPA에 연결하는 CRD + 컨트롤러입니다. `ScaledObject`를 정의합니다:

```yaml
kind: ScaledObject
metadata: { name: rabbitmq-consumer }
spec:
  scaleTargetRef: { name: my-consumer }
  minReplicaCount: 0
  maxReplicaCount: 100
  triggers:
    - type: rabbitmq
      metadata:
        host: amqp://...
        queueName: jobs
        queueLength: "5"     # 메시지 5개당 1개 파드
```

KEDA는 RabbitMQ를 폴링하고, `desiredReplicas = ceil(queueLength / 5)`를 계산하고, 이를 생성된 HPA에 노출합니다. 결정적 추가 능력 — **scale to zero**. HPA는 1 미만으로 스케일할 수 없습니다 — KEDA는 큐가 비어 있을 때 0으로 스케일하고 메시지가 도착하면 다시 위로(operator 패턴을 통해 첫 파드 생성). 이는 폭발적 워크로드에서 비용에 대해 거대합니다 — 작업이 실제 실행 중인 시간만 지불합니다.

KEDA는 60+ 스케일러(RabbitMQ, Kafka, AWS SQS, Postgres 쿼리, Cron, Prometheus, ...)를 가집니다. 부하가 CPU 모양이 아니라 이벤트 모양인 워크로드에서, KEDA는 HPA-on-custom-metrics를 훨씬 단순한 구성으로 대체합니다.

### 이론에서 아래의 YAML으로

이제 레슨은 이 추상을 적용합니다:

- **섹션 1 (HPA v2)**는 §A입니다 — 공식, 구성, behavior 정책.
- **섹션 2 (커스텀과 외부 메트릭)**은 §B입니다 — Prometheus Adapter, 커스텀 어댑터, 비즈니스 KPI.
- **섹션 3 (VPA)**는 §C입니다 — 모드와 운영 사용.
- **섹션 4 (Cluster Autoscaler)**는 클라우드 프로바이더 통합과 함께한 §C의 노드 수준 루프입니다.
- **섹션 5 (KEDA)**는 §D입니다 — `ScaledObject`, 스케일러, scale-to-zero.
- **섹션 6 (Prometheus로 스케일링)**은 §B의 어댑터 패턴을 실세계 신호와 결합합니다.
- **섹션 7 (예측 오토스케일링)**은 반응적(P-control)에서 예측적(이력으로부터 다음 N분 예측)으로 이동합니다.
- **섹션 8 (비용 인식 스케일링)**은 spot/on-demand 믹스와 노드 그룹 선택을 스케일링 결정에 통합합니다.
- **섹션 9 (모범 사례)**는 서로 싸우지 않으면서 §A와 §C의 세 계층을 운영적으로 합성하는 것입니다.

HPA / VPA / CA를 다른 신호와 액추에이터를 가진 세 개의 독립 제어 루프로 보고 나면, "왜 내 파드가 스케일 업하지 않는가?"는 "어느 루프, 어느 메트릭, 어느 lag?"으로 환원됩니다.

---

## 1. Horizontal Pod Autoscaler (HPA) v2

### 1.1 HPA 작동 방식

HPA 컨트롤러는 제어 루프(기본 15초마다)를 실행하여 다음을 수행합니다:

1. 메트릭 API에서 메트릭 조회
2. 원하는 레플리카 수 계산
3. 대상 워크로드 스케일링

```
                          ┌─────────────────────┐
                          │    Metrics Server    │
                          │  (or custom adapter) │
                          └──────────┬──────────┘
                                     │ metrics
                                     ▼
┌─────────────┐          ┌─────────────────────┐         ┌─────────────────┐
│  Target      │◀─────────│   HPA Controller    │────────▶│  Scale          │
│  Workload    │  observe  │                     │  scale   │  Subresource    │
│  (Deployment)│          │  desiredReplicas =   │         │  /scale         │
└─────────────┘          │  ceil(current *      │         └─────────────────┘
                          │   currentMetric /    │
                          │   desiredMetric)     │
                          └─────────────────────┘
```

### 1.2 HPA v2 API

`autoscaling/v2` API는 여러 메트릭 유형을 동시에 지원합니다:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 3
  maxReplicas: 50
  metrics:
  # Resource metric (CPU)
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Resource metric (memory) -- absolute value
  - type: Resource
    resource:
      name: memory
      target:
        type: AverageValue
        averageValue: 500Mi
  # Pod metric (custom metric from application)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  # Object metric (from another Kubernetes object)
  - type: Object
    object:
      describedObject:
        apiVersion: networking.k8s.io/v1
        kind: Ingress
        name: web-app-ingress
      metric:
        name: requests_per_second
      target:
        type: Value
        value: "10000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 4
        periodSeconds: 60
      - type: Percent
        value: 100
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 120
      selectPolicy: Min
```

### 1.3 스케일링 공식

```
desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]
```

여러 메트릭이 지정된 경우, HPA는 각 메트릭에 대해 원하는 레플리카 수를 계산하고 **최댓값**을 취합니다:

```
finalReplicas = max(desiredFromCPU, desiredFromMemory, desiredFromCustom)
```

예시:
- 현재 레플리카: 5
- 현재 CPU 사용률: 90%
- 목표 CPU 사용률: 70%
- 원하는 값 = ceil(5 * 90/70) = ceil(6.43) = 7

### 1.4 스케일링 동작

`behavior` 필드는 스케일링의 속도와 안정성을 제어합니다:

```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 0     # Scale up immediately
    policies:
    - type: Pods
      value: 10                        # Add max 10 pods per period
      periodSeconds: 60
    - type: Percent
      value: 200                       # Or double pods per period
      periodSeconds: 60
    selectPolicy: Max                  # Use the larger of the two

  scaleDown:
    stabilizationWindowSeconds: 300    # Wait 5 min of sustained low metrics
    policies:
    - type: Pods
      value: 2                         # Remove max 2 pods per period
      periodSeconds: 120
    selectPolicy: Min                  # Use the smaller value (conservative)
```

### 1.5 전제 조건

```bash
# HPA requires metrics-server for resource metrics (CPU/memory)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify metrics-server is running
kubectl get deployment metrics-server -n kube-system
kubectl top pods
kubectl top nodes
```

### 1.6 HPA 명령어

```bash
# Create a simple HPA
kubectl autoscale deployment web-app --cpu-percent=70 --min=3 --max=50

# View HPA status
kubectl get hpa web-app-hpa
# NAME          REFERENCE            TARGETS          MINPODS   MAXPODS   REPLICAS   AGE
# web-app-hpa   Deployment/web-app   45%/70%, 350Mi/500Mi   3         50        5          2h

# Describe HPA for detailed status and events
kubectl describe hpa web-app-hpa

# Check HPA conditions
kubectl get hpa web-app-hpa -o jsonpath='{.status.conditions[*].type}'
# AbleToScale ScalingActive ScalingLimited
```

---

## 2. 커스텀 메트릭과 외부 메트릭

### 2.1 메트릭 API 아키텍처

```
                    ┌──────────────────────┐
                    │    HPA Controller     │
                    └────────┬─────────────┘
                             │ queries
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
    │ metrics.k8s.io│ │custom.metrics│ │external.metrics│
    │ (Resource)    │ │.k8s.io      │ │.k8s.io         │
    └──────┬───────┘ └──────┬───────┘ └──────┬─────────┘
           │                │                │
           ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
    │metrics-server│ │  Prometheus  │ │  External API  │
    │              │ │  Adapter     │ │  Adapter       │
    └──────────────┘ └──────────────┘ └────────────────┘
```

### 2.2 메트릭 유형

| 유형 | API 그룹 | 소스 | 예시 |
|---|---|---|---|
| Resource | `metrics.k8s.io` | kubelet cAdvisor | Pod별 CPU, 메모리 |
| Pod | `custom.metrics.k8s.io` | 어댑터를 통한 애플리케이션 메트릭 | `http_requests_per_second` |
| Object | `custom.metrics.k8s.io` | K8s 객체의 메트릭 | Ingress 요청 비율 |
| External | `external.metrics.k8s.io` | 클러스터 외부의 메트릭 | SQS 큐 깊이, Pub/Sub 백로그 |

### 2.3 Prometheus 어댑터

Prometheus 어댑터는 Prometheus 메트릭을 Kubernetes 커스텀 메트릭 API로 연결합니다:

```bash
# Install Prometheus adapter
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --set prometheus.url=http://prometheus-server.monitoring.svc \
  --set prometheus.port=9090
```

애플리케이션 메트릭을 노출하는 어댑터 구성:

```yaml
# prometheus-adapter-config.yaml
rules:
# Map Prometheus http_requests_total to custom.metrics.k8s.io
- seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    matches: "^(.*)_total$"
    as: "${1}_per_second"
  metricsQuery: 'rate(<<.Series>>{<<.LabelMatchers>>}[2m])'

# Map queue depth
- seriesQuery: 'rabbitmq_queue_messages{namespace!="",service!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      service: {resource: "service"}
  name:
    as: "queue_messages"
  metricsQuery: '<<.Series>>{<<.LabelMatchers>>}'
```

### 2.4 커스텀 메트릭 확인

```bash
# List available custom metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq '.resources[].name'

# Query a specific metric
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/production/pods/*/http_requests_per_second" | jq .

# List external metrics
kubectl get --raw /apis/external.metrics.k8s.io/v1beta1 | jq '.resources[].name'
```

---

## 3. Vertical Pod Autoscaler (VPA)

### 3.1 VPA란?

VPA는 관찰된 사용량을 기반으로 컨테이너의 CPU 및 메모리 **요청(request)과 제한(limit)**을 자동으로 조정합니다. 더 많은 Pod를 추가하는 대신 기존 Pod의 적정 규모를 조정합니다.

### 3.2 VPA 구성 요소

```
┌─────────────────────────────────────────────────┐
│                VPA System                        │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Recommender │  │     Admission Controller │  │
│  │  (analyzes    │  │  (applies recommendations│  │
│  │   metrics)    │  │   at pod creation)       │  │
│  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                      │                  │
│         ▼                      ▼                  │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Updater    │  │     VPA Object           │  │
│  │  (evicts pods │  │  (stores recommendations │  │
│  │   for resize) │  │   and policy)            │  │
│  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 3.3 설치

```bash
# Clone and install VPA
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh

# Verify
kubectl get pods -n kube-system | grep vpa
```

### 3.4 VPA 구성

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: web-app-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  updatePolicy:
    updateMode: Auto   # Off, Initial, Recreate, or Auto
  resourcePolicy:
    containerPolicies:
    - containerName: web-app
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 4
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits  # or RequestsOnly
    - containerName: sidecar
      mode: "Off"  # Do not adjust sidecar resources
```

### 3.5 업데이트 모드

| 모드 | 동작 | 사용 사례 |
|---|---|---|
| `Off` | 권장 사항만 제공, 변경 없음 | 관찰 단계 |
| `Initial` | Pod 생성 시에만 리소스 설정 | 안정적인 워크로드, 재시작 방지 |
| `Recreate` | Pod를 퇴거(evict)하고 재생성하여 변경 적용 | 일반 사용 |
| `Auto` | 현재 Recreate와 동일, 향후 인플레이스(in-place) 지원 가능 | 권장 기본값 |

### 3.6 VPA 권장 사항 읽기

```bash
# Get VPA recommendations
kubectl describe vpa web-app-vpa

# Example output:
# Recommendation:
#   Container Recommendations:
#     Container Name: web-app
#     Lower Bound:
#       Cpu:     100m
#       Memory:  256Mi
#     Target:
#       Cpu:     350m
#       Memory:  512Mi
#     Uncapped Target:
#       Cpu:     350m
#       Memory:  512Mi
#     Upper Bound:
#       Cpu:     2
#       Memory:  2Gi
```

### 3.7 VPA와 HPA 상호 작용

**동일한 메트릭(CPU)에서 VPA와 HPA를 함께 사용하지 마세요.** 충돌이 발생합니다:

| 조합 | 작동? | 참고 |
|---|---|---|
| HPA CPU + VPA CPU | 아니오 | 둘 다 CPU를 제어하려 하여 진동 발생 |
| HPA 커스텀 메트릭 + VPA CPU/메모리 | 예 | 서로 다른 제어 차원 |
| HPA + VPA `Off` 모드 | 예 | VPA는 권장만 하고 작동하지 않음 |

---

## 4. Cluster Autoscaler

### 4.1 Cluster Autoscaler란?

HPA와 VPA가 워크로드를 조정하는 반면, Cluster Autoscaler는 클러스터의 **노드** 수를 조정합니다. Pod가 스케줄링 불가능할 때 노드를 추가하고 활용도가 낮은 노드를 제거합니다.

### 4.2 스케일 업 결정

```
Pod pending (unschedulable)
    │
    ▼
Cluster Autoscaler checks:
    1. Is there a node group that could fit this pod?
    2. Would adding a node make the pod schedulable?
    3. Is the node group below its max size?
    │
    ▼ (all yes)
Request new node from cloud provider
    │
    ▼
Wait for node to join the cluster (1-5 min)
    │
    ▼
Scheduler places pending pod on new node
```

### 4.3 스케일 다운 결정

```
Node utilization check (every 10s)
    │
    ▼
Is node utilization < 50% (default)?
    │ yes
    ▼
Can all pods be moved to other nodes?
    │ yes
    ▼
Are there any blocking conditions?
    - PDB would be violated?
    - Pod with local storage?
    - Pod without controller (bare pod)?
    - Pod with "cluster-autoscaler.kubernetes.io/safe-to-evict: false"?
    │ no blockers
    ▼
Wait 10 minutes (scale-down-unneeded-time)
    │
    ▼
Drain and delete node
```

### 4.4 설치 (AWS EKS 예제)

```bash
# Install Cluster Autoscaler on EKS
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set autoDiscovery.clusterName=my-cluster \
  --set awsRegion=us-west-2 \
  --set extraArgs.balance-similar-node-groups=true \
  --set extraArgs.skip-nodes-with-local-storage=false \
  --set extraArgs.expander=least-waste \
  --set extraArgs.scale-down-utilization-threshold=0.5 \
  --set extraArgs.scale-down-unneeded-time=10m \
  --set extraArgs.scale-down-delay-after-add=10m
```

### 4.5 노드 그룹 구성

```yaml
# AWS Auto Scaling Group tags for auto-discovery
# k8s.io/cluster-autoscaler/enabled: true
# k8s.io/cluster-autoscaler/my-cluster: owned

# Priority-based expander configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: kube-system
data:
  priorities: |
    50:
    - name: spot-node-group.*
    30:
    - name: on-demand-node-group.*
    10:
    - name: gpu-node-group.*
```

### 4.6 Cluster Autoscaler를 위한 Pod 어노테이션

```yaml
metadata:
  annotations:
    # Tell CA this pod is safe to evict (for scale-down)
    cluster-autoscaler.kubernetes.io/safe-to-evict: "true"

    # Tell CA this pod is NOT safe to evict
    cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
```

---

## 5. KEDA (Kubernetes Event-Driven Autoscaling)

### 5.1 KEDA란?

KEDA는 Kubernetes 오토스케일링을 이벤트 기반 워크로드로 확장합니다. 메시지 큐, 데이터베이스, cron 스케줄, 클라우드 서비스와 같은 이벤트 소스를 기반으로 스케일링할 수 있습니다.

### 5.2 아키텍처

```
              ┌──────────────────────────┐
              │     Event Sources         │
              │  (RabbitMQ, Kafka, SQS,  │
              │   Redis, Prometheus...)   │
              └────────────┬─────────────┘
                           │ poll
                           ▼
              ┌──────────────────────────┐
              │      KEDA Operator       │
              │  ┌────────────────────┐  │
              │  │   Metrics Server   │  │  expose metrics
              │  │   (custom metrics) │──────────────┐
              │  └────────────────────┘  │            │
              │  ┌────────────────────┐  │            ▼
              │  │   Controller       │  │     ┌─────────────┐
              │  │  (scale to/from 0) │──────▶│     HPA     │
              │  └────────────────────┘  │     └─────────────┘
              └──────────────────────────┘            │
                                                      ▼
                                              ┌───────────────┐
                                              │  Deployment   │
                                              │  (0 → N pods) │
                                              └───────────────┘
```

### 5.3 설치

```bash
# Install KEDA using Helm
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --set watchNamespace="" \
  --set operator.replicaCount=2
```

### 5.4 큐 기반 스케일링을 위한 ScaledObject

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: order-processor
  namespace: production
spec:
  scaleTargetRef:
    name: order-processor
  pollingInterval: 15           # Check every 15 seconds
  cooldownPeriod: 300           # Wait 5 min before scale-to-zero
  idleReplicaCount: 0           # Scale to zero when idle
  minReplicaCount: 0            # Minimum replicas
  maxReplicaCount: 100          # Maximum replicas
  fallback:
    failureThreshold: 3
    replicas: 5                  # Fallback replica count if scaler fails
  triggers:
  - type: rabbitmq
    metadata:
      host: amqp://guest:guest@rabbitmq.default.svc:5672/
      queueName: orders
      queueLength: "10"          # 1 pod per 10 messages
    authenticationRef:
      name: rabbitmq-credentials
```

### 5.5 Kafka용 ScaledObject

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: kafka-consumer
  namespace: production
spec:
  scaleTargetRef:
    name: kafka-consumer
  minReplicaCount: 1
  maxReplicaCount: 30
  triggers:
  - type: kafka
    metadata:
      bootstrapServers: kafka-broker.kafka.svc:9092
      consumerGroup: order-group
      topic: orders
      lagThreshold: "100"         # Scale when lag > 100 per partition
      offsetResetPolicy: latest
      allowIdleConsumers: "false"
      scaleToZeroOnInvalidOffset: "false"
```

### 5.6 Cron 트리거를 사용한 ScaledObject

```yaml
# Predictive scaling based on schedule
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: web-frontend
  namespace: production
spec:
  scaleTargetRef:
    name: web-frontend
  minReplicaCount: 3
  maxReplicaCount: 100
  triggers:
  # Prometheus-based reactive scaling
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server.monitoring.svc:9090
      metricName: http_requests_per_second
      query: sum(rate(http_requests_total{deployment="web-frontend"}[2m]))
      threshold: "500"

  # Cron-based proactive scaling for known traffic patterns
  - type: cron
    metadata:
      timezone: America/New_York
      start: 0 8 * * 1-5          # Mon-Fri 8:00 AM
      end: 0 20 * * 1-5           # Mon-Fri 8:00 PM
      desiredReplicas: "20"        # Pre-scale for business hours
```

### 5.7 배치 처리를 위한 ScaledJob

```yaml
# KEDA can also scale Jobs (not just Deployments)
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: image-processor
  namespace: production
spec:
  jobTargetRef:
    template:
      spec:
        containers:
        - name: processor
          image: example.com/image-processor:v1
          env:
          - name: QUEUE_URL
            value: "https://sqs.us-west-2.amazonaws.com/123456/images"
        restartPolicy: Never
    backoffLimit: 3
  pollingInterval: 10
  maxReplicaCount: 50
  successfulJobsHistoryLimit: 10
  failedJobsHistoryLimit: 5
  scalingStrategy:
    strategy: accurate
  triggers:
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.us-west-2.amazonaws.com/123456/images
      queueLength: "5"
      awsRegion: us-west-2
    authenticationRef:
      name: aws-credentials
```

---

## 6. Prometheus 메트릭을 이용한 스케일링

### 6.1 HPA용 애플리케이션 메트릭

HPA 소비를 위해 애플리케이션에서 메트릭을 노출합니다:

```go
// Go application with Prometheus metrics
package main

import (
    "net/http"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    httpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "path", "status"},
    )
    httpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "path"},
    )
    activeConnections = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "active_connections",
            Help: "Number of active connections",
        },
    )
    queueDepth = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "queue_depth",
            Help: "Number of items in the processing queue",
        },
    )
)

func main() {
    http.Handle("/metrics", promhttp.Handler())
    http.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
        activeConnections.Inc()
        defer activeConnections.Dec()
        // handler logic
        httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, "200").Inc()
    })
    http.ListenAndServe(":8080", nil)
}
```

### 6.2 Prometheus 어댑터 규칙

```yaml
# Map application metrics to HPA-consumable metrics
rules:
- seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    matches: "^(.*)_total$"
    as: "${1}_per_second"
  metricsQuery: 'sum(rate(<<.Series>>{<<.LabelMatchers>>}[2m])) by (<<.GroupBy>>)'

- seriesQuery: 'active_connections{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    as: "active_connections"
  metricsQuery: '<<.Series>>{<<.LabelMatchers>>}'

- seriesQuery: 'queue_depth{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    as: "queue_depth"
  metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

### 6.3 Prometheus 메트릭을 사용하는 HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "500"
  - type: Pods
    pods:
      metric:
        name: active_connections
      target:
        type: AverageValue
        averageValue: "100"
```

---

## 7. 예측 오토스케일링

### 7.1 반응형 스케일링의 문제

반응형 오토스케일링(기존 HPA)에는 고유한 지연 시간이 있습니다:

```
Traffic spike arrives
    │
    ▼  (15s) HPA scrape interval
HPA detects increased metric
    │
    ▼  (seconds) HPA computation
Scale decision made
    │
    ▼  (30-120s) Pod scheduling + startup
New pods ready to serve
    │
    Total delay: 45s - 3 min
```

예측 가능한 트래픽 패턴을 가진 워크로드의 경우, 이 지연은 증가 기간 동안 성능 저하를 야기합니다.

### 7.2 KEDA Cron 기반 사전 스케일링

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: predictive-web
spec:
  scaleTargetRef:
    name: web-frontend
  minReplicaCount: 3
  maxReplicaCount: 100
  triggers:
  # Reactive: handle unexpected spikes
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring.svc:9090
      metricName: http_rps
      query: sum(rate(http_requests_total{app="web-frontend"}[2m]))
      threshold: "500"

  # Predictive: pre-scale for known patterns
  - type: cron
    metadata:
      timezone: UTC
      start: 30 7 * * 1-5      # Pre-scale at 7:30 AM weekdays
      end: 0 9 * * 1-5          # Hold until 9 AM
      desiredReplicas: "30"

  - type: cron
    metadata:
      timezone: UTC
      start: 30 11 * * 1-5     # Lunch spike
      end: 0 14 * * 1-5
      desiredReplicas: "25"

  - type: cron
    metadata:
      timezone: UTC
      start: 0 0 25 11 *       # Black Friday
      end: 0 0 27 11 *
      desiredReplicas: "80"
```

### 7.3 HPA와 PodDisruptionBudget 결합

스케일 다운으로 인한 서비스 중단을 방지합니다:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-app-pdb
spec:
  minAvailable: "80%"
  selector:
    matchLabels:
      app: web-app
```

---

## 8. 비용 인식 스케일링

### 8.1 VPA 권장 사항을 활용한 적정 규모 조정

```bash
# Use VPA in Off mode to collect recommendations without applying them
kubectl get vpa web-app-vpa -o jsonpath='{.status.recommendation.containerRecommendations[0]}' | jq .

# Use goldilocks to get VPA recommendations for all workloads
# https://github.com/FairwindsOps/goldilocks
helm install goldilocks fairwinds-stable/goldilocks --namespace goldilocks --create-namespace
```

### 8.2 스팟/선점형 인스턴스

```yaml
# Node affinity for cost-aware scheduling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: batch-processor
spec:
  replicas: 10
  template:
    spec:
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 90
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: ["spot"]
          - weight: 10
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: ["on-demand"]
      tolerations:
      - key: "kubernetes.io/spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: processor
        image: example.com/batch-processor:v1
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: "1"
            memory: 1Gi
```

### 8.3 Cluster Autoscaler 우선순위 확장기

저렴한 노드 그룹을 우선하도록 Cluster Autoscaler를 구성합니다:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: kube-system
data:
  priorities: |
    100:
    - name: spot-.*          # Highest priority: spot instances
    50:
    - name: on-demand-small-.*  # Medium: small on-demand
    10:
    - name: on-demand-large-.*  # Last resort: large on-demand
```

### 8.4 스케일 다운 최적화

```bash
# Aggressive scale-down for non-production
--scale-down-unneeded-time=3m
--scale-down-utilization-threshold=0.3
--scale-down-delay-after-add=3m

# Conservative scale-down for production
--scale-down-unneeded-time=15m
--scale-down-utilization-threshold=0.5
--scale-down-delay-after-add=15m
--scale-down-delay-after-failure=5m
```

---

## 9. 스케일링 모범 사례

### 9.1 리소스 요청이 핵심

HPA는 사용률을 리소스 **요청(request)**의 백분율로 계산하며, 제한(limit)이 아닙니다. 요청을 너무 높게 설정하면 스케일링이 부족하고, 너무 낮게 설정하면 과도한 스케일링이 발생합니다.

```yaml
# BAD: requests too high, HPA will never trigger
resources:
  requests:
    cpu: "4"        # Using 300m of 4 cores = 7.5% utilization
  limits:
    cpu: "4"

# GOOD: requests reflect actual steady-state usage
resources:
  requests:
    cpu: 500m       # Using 300m of 500m = 60% utilization
  limits:
    cpu: "2"
```

### 9.2 스케일링 체크리스트

| 관행 | 권장 사항 |
|---|---|
| 모든 컨테이너에 리소스 요청 설정 | HPA CPU/메모리 메트릭에 필수 |
| readiness 프로브 사용 | 준비되지 않은 Pod로 트래픽 라우팅 방지 |
| PodDisruptionBudget 사용 | 스케일 다운으로 인한 서비스 중단 방지 |
| 보수적 동작으로 시작 | 긴 안정화 윈도우, 느린 스케일 다운 |
| HPA 결정 모니터링 | `kubectl describe hpa`로 스케일링 이벤트 확인 |
| 동일 메트릭에서 HPA + VPA 피하기 | 커스텀 메트릭의 HPA + 리소스의 VPA 사용 |
| maxReplicas를 신중하게 설정 | 노드 용량과 비용 예산 고려 |
| 부하 테스트로 스케일링 테스트 | k6 또는 Locust 같은 도구로 부하 테스트 |

### 9.3 오토스케일링 부하 테스트

```bash
# Use k6 to test autoscaling behavior
cat > load-test.js <<'EOF'
import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 500 },   // Spike to 500 users
    { duration: '5m', target: 500 },   // Stay at 500 users
    { duration: '5m', target: 0 },     // Ramp down
  ],
};

export default function () {
  http.get('http://web-app.production.svc/api/health');
  sleep(1);
}
EOF

kubectl run k6 --image=grafana/k6 --rm -it --restart=Never -- run - < load-test.js
```

부하 테스트 중 모니터링:

```bash
# Watch HPA in real-time
kubectl get hpa -w

# Watch pod count
kubectl get pods -l app=web-app -w

# Watch node count
kubectl get nodes -w
```

---

## 연습문제

### 연습문제 1: 다중 메트릭 HPA

`api-server`라는 이름의 웹 애플리케이션 Deployment를 위해 세 가지 메트릭을 동시에 기반으로 스케일링하는 HPA를 생성하세요: (a) CPU 사용률 목표 60%, (b) 메모리 사용률 목표 75%, (c) Pod당 평균값 200의 `http_requests_per_second` 커스텀 메트릭. 최소 레플리카를 2, 최대를 30으로 설정하세요. 스케일 업은 분당 최대 5개 Pod를 추가하고, 스케일 다운은 2분마다 최대 1개 Pod를 5분 안정화 윈도우와 함께 제거하도록 동작을 구성하세요.

<details>
<summary>정답 보기</summary>

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 2
  maxReplicas: 30
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "200"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 5
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
      selectPolicy: Min
```

검증:

```bash
kubectl apply -f api-server-hpa.yaml
kubectl get hpa api-server-hpa
kubectl describe hpa api-server-hpa
```

</details>

### 연습문제 2: VPA 구성

예측 불가능한 메모리 사용량을 가진 Java 애플리케이션에 대해 VPA를 구성하세요. Deployment에는 `app` (Java)과 `envoy-proxy` (사이드카) 두 개의 컨테이너가 있습니다. 요구사항: (a) VPA는 `app` 컨테이너만 관리, (b) CPU는 250m에서 4코어 사이, (c) 메모리는 512Mi에서 8Gi 사이, (d) 사이드카는 수정하지 않음, (e) 기존 Pod가 재시작되지 않도록 `Initial` 모드 사용. 현재 권장 사항을 확인하는 명령어도 작성하세요.

<details>
<summary>정답 보기</summary>

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: java-app-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: java-app
  updatePolicy:
    updateMode: Initial
  resourcePolicy:
    containerPolicies:
    - containerName: app
      minAllowed:
        cpu: 250m
        memory: 512Mi
      maxAllowed:
        cpu: "4"
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
    - containerName: envoy-proxy
      mode: "Off"
```

권장 사항 확인:

```bash
# View current VPA recommendation
kubectl get vpa java-app-vpa -n production -o yaml | \
  yq '.status.recommendation.containerRecommendations'

# Or with kubectl describe
kubectl describe vpa java-app-vpa -n production

# Check that only the 'app' container is being recommended
kubectl get vpa java-app-vpa -n production \
  -o jsonpath='{range .status.recommendation.containerRecommendations[*]}{.containerName}: CPU={.target.cpu}, Memory={.target.memory}{"\n"}{end}'
```

</details>

### 연습문제 3: 다중 트리거 KEDA

다음을 기반으로 스케일링하는 `order-processor` Deployment용 KEDA ScaledObject를 생성하세요: (a) RabbitMQ 큐 깊이 (`orders` 큐에서 5개 메시지당 1 Pod), (b) 업무 시간(EST 오전 9시 - 오후 6시, 평일) 동안 10개 레플리카로 사전 스케일링하는 cron 스케줄, (c) 처리 지연이 500ms를 초과할 때 스케일링하는 Prometheus 메트릭 폴백. 시스템은 비업무 시간에 5분 쿨다운으로 제로 스케일링해야 합니다. RabbitMQ용 TriggerAuthentication을 포함하세요.

<details>
<summary>정답 보기</summary>

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rabbitmq-secret
  namespace: production
data:
  host: YW1xcDovL3VzZXI6cGFzc0ByYWJiaXRtcS5wcm9kdWN0aW9uLnN2Yzo1NjcyLw==
---
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: rabbitmq-auth
  namespace: production
spec:
  secretTargetRef:
  - parameter: host
    name: rabbitmq-secret
    key: host
---
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: order-processor
  namespace: production
spec:
  scaleTargetRef:
    name: order-processor
  pollingInterval: 10
  cooldownPeriod: 300
  idleReplicaCount: 0
  minReplicaCount: 0
  maxReplicaCount: 50
  fallback:
    failureThreshold: 3
    replicas: 5
  triggers:
  # RabbitMQ queue depth
  - type: rabbitmq
    metadata:
      queueName: orders
      queueLength: "5"
      protocol: amqp
    authenticationRef:
      name: rabbitmq-auth

  # Cron: pre-scale during business hours
  - type: cron
    metadata:
      timezone: America/New_York
      start: 0 9 * * 1-5
      end: 0 18 * * 1-5
      desiredReplicas: "10"

  # Prometheus: scale on processing latency
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server.monitoring.svc:9090
      metricName: order_processing_latency
      query: |
        histogram_quantile(0.95,
          rate(order_processing_duration_seconds_bucket{deployment="order-processor"}[5m])
        )
      threshold: "0.5"
      activationThreshold: "0.3"
```

검증:

```bash
kubectl get scaledobject order-processor -n production
kubectl get hpa -n production  # KEDA creates an HPA under the hood
kubectl describe scaledobject order-processor -n production
```

</details>

### 연습문제 4: Cluster Autoscaler 트러블슈팅

클러스터에 3개의 노드 그룹(spot-small, spot-large, on-demand)이 있습니다. Pod가 pending 상태이지만 Cluster Autoscaler가 스케일 업하지 않습니다. 다음을 수행하는 명령어를 작성하세요: (a) Cluster Autoscaler 상태 ConfigMap 확인, (b) 스케일 업 결정에 대한 CA 로그 확인, (c) pending Pod에 배치를 방해하는 스케줄링 제약 조건이 있는지 확인, (d) 노드 그룹 구성 확인. 그런 다음 CA가 스케일 업에 실패하는 세 가지 일반적인 이유와 해결책을 설명하세요.

<details>
<summary>정답 보기</summary>

```bash
# (a) Check CA status ConfigMap
kubectl get configmap cluster-autoscaler-status -n kube-system -o yaml

# (b) View CA logs for scale-up decisions
kubectl logs -n kube-system -l app.kubernetes.io/name=cluster-autoscaler --tail=200 | grep -E "Scale|scale|pending|unschedulable"

# (c) Check pending pod details
kubectl get pods --field-selector=status.phase=Pending -A
kubectl describe pod <pending-pod> -n <namespace>
# Look for Events section, especially "FailedScheduling" with reason

# (d) Verify node group configuration
kubectl get nodes --show-labels | grep node.kubernetes.io/instance-type
kubectl get nodes -o custom-columns="NAME:.metadata.name,CAPACITY_CPU:.status.capacity.cpu,CAPACITY_MEM:.status.capacity.memory,ALLOCATABLE_CPU:.status.allocatable.cpu"
```

**CA가 스케일 업에 실패하는 세 가지 일반적인 이유:**

1. **리소스 요청이 노드 용량 초과**: pending Pod가 노드 그룹의 어떤 노드보다 더 많은 CPU/메모리를 요청합니다. 해결책: 더 큰 인스턴스 유형의 노드 그룹을 생성하거나 Pod의 리소스 요청을 줄이세요.

2. **노드 그룹이 최대 크기에 도달**: Pod를 수용할 수 있는 노드 그룹이 이미 구성된 최대값에 도달했습니다. 해결책: 노드 그룹의 `maxSize`를 늘리거나 추가 노드 그룹을 생성하세요.

3. **Pod에 충족할 수 없는 제약 조건**: Pod가 어떤 노드 그룹도 충족할 수 없는 `nodeSelector`, `nodeAffinity` 또는 `tolerations`를 지정합니다. 해결책: 노드 그룹 시작 템플릿에 레이블을 추가하거나 Pod의 toleration과 일치하는 taint를 추가하세요. 다음으로 확인:

```bash
# Check what affinity/selectors the pod requires
kubectl get pod <pending-pod> -o jsonpath='{.spec.nodeSelector}'
kubectl get pod <pending-pod> -o jsonpath='{.spec.affinity}'
kubectl get pod <pending-pod> -o jsonpath='{.spec.tolerations}'

# Compare with available node labels
kubectl get nodes --show-labels
```

</details>

### 연습문제 5: 엔드투엔드 오토스케일링 설계

세 가지 티어를 가진 마이크로서비스 애플리케이션을 위한 오토스케일링 전략을 설계하세요: (a) API 게이트웨이 (지연에 민감, 100ms 미만으로 응답해야 함), (b) 백그라운드 잡 프로세서 (SQS 큐에서 소비, 30초 지연 허용 가능), (c) 데이터 파이프라인 (야간 실행, 비용에 민감). 각 티어에 대해 오토스케일러 유형(HPA/VPA/KEDA/CA), 스케일링할 메트릭, 최소/최대 레플리카, 스케일링 동작, 노드 유형(온디맨드 vs 스팟)을 지정하세요. 세 티어 모두에 대한 HPA/KEDA 매니페스트를 작성하세요.

<details>
<summary>정답 보기</summary>

**설계:**

| 티어 | 오토스케일러 | 메트릭 | 최소/최대 | 노드 유형 |
|---|---|---|---|---|
| API 게이트웨이 | HPA v2 | CPU (60%), p99 지연 (<80ms) | 5/100 | 온디맨드 |
| 잡 프로세서 | KEDA | SQS 큐 깊이 | 0/50 | 스팟 |
| 데이터 파이프라인 | KEDA (cron + ScaledJob) | Cron 스케줄 | 0/30 | 스팟 |

```yaml
# (a) API Gateway - HPA with aggressive scale-up
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 5
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Pods
    pods:
      metric:
        name: http_request_duration_p99
      target:
        type: AverageValue
        averageValue: "80m"  # 80ms target, scale before hitting 100ms SLO
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Percent
        value: 100                    # Double pods instantly if needed
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
---
# (b) Job Processor - KEDA with SQS
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: job-processor
  namespace: production
spec:
  scaleTargetRef:
    name: job-processor
  pollingInterval: 15
  cooldownPeriod: 300
  idleReplicaCount: 0
  minReplicaCount: 0
  maxReplicaCount: 50
  triggers:
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.us-west-2.amazonaws.com/123456/jobs
      queueLength: "10"
      awsRegion: us-west-2
    authenticationRef:
      name: aws-credentials
---
# (c) Data Pipeline - KEDA ScaledJob with cron
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: data-pipeline
  namespace: batch
spec:
  jobTargetRef:
    template:
      spec:
        containers:
        - name: pipeline
          image: example.com/data-pipeline:v1
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
        restartPolicy: Never
        tolerations:
        - key: "kubernetes.io/spot"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
        nodeSelector:
          node.kubernetes.io/instance-type: spot
    backoffLimit: 2
  pollingInterval: 30
  maxReplicaCount: 30
  successfulJobsHistoryLimit: 5
  failedJobsHistoryLimit: 3
  triggers:
  - type: cron
    metadata:
      timezone: UTC
      start: 0 2 * * *      # Start at 2 AM UTC
      end: 0 6 * * *         # End at 6 AM UTC
      desiredReplicas: "20"
```

</details>

---

**이전**: [어드미션 컨트롤러](./12_Admission_Controllers.md) | **다음**: [관측 가능성](./14_Observability.md)
