# 레슨 9: 서비스 메시와 네트워킹

**이전**: [컨테이너 오케스트레이션 운영](./08_Container_Orchestration_Operations.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 서비스 메시 패턴과 마이크로서비스 네트워킹 복잡성을 처리하기 위해 등장한 이유를 설명할 수 있습니다
2. 컨트롤 플레인(istiod)과 데이터 플레인(Envoy 사이드카)을 포함한 Istio 아키텍처를 설명할 수 있습니다
3. 카나리 배포, 트래픽 분할, 재시도, 서킷 브레이킹을 포함한 트래픽 관리 정책을 구성할 수 있습니다
4. 제로 트러스트(zero-trust) 서비스 간 통신을 위한 상호 TLS(mTLS)를 구현할 수 있습니다
5. Istio의 관찰 가능성 기능을 사용하여 서비스 트래픽, 지연 시간, 오류율에 대한 가시성을 확보할 수 있습니다

---

애플리케이션이 모놀리스에서 수십 또는 수백 개의 마이크로서비스로 진화함에 따라, 서비스 간 네트워킹은 핵심적인 관심사가 됩니다. 각 서비스는 로드밸런싱, 재시도, 타임아웃, 서킷 브레이킹, 상호 인증, 관찰 가능성이 필요하며 -- 이를 모든 서비스의 애플리케이션 코드에 구현하면 엄청난 중복과 비일관성이 발생합니다. 서비스 메시는 네트워킹 로직을 애플리케이션에서 분리하여 전용 인프라 계층으로 이동시킴으로써 이 문제를 해결합니다. 이 레슨에서는 가장 널리 채택된 서비스 메시인 Istio를 중심으로 서비스 메시 개념을 다룹니다.

> **비유 -- 우편 시스템**: 서비스 메시가 없으면 모든 마이크로서비스는 모든 편지를 직접 배달해야 하는 사람과 같습니다: 주소를 찾고, 경로를 선택하고, 실패를 처리하고, 배달을 추적해야 합니다. 서비스 메시는 우편 시스템과 같습니다: 각 서비스는 메시지를 우편함(사이드카 프록시)에 넣고, 우편 인프라가 라우팅, 배달 확인, 실패 시 재시도, 추적을 처리합니다 -- 발신자가 세부 사항을 알 필요가 없습니다.

## 1. 서비스 메시가 필요한 이유

### 마이크로서비스 네트워킹 문제

```
모놀리스:
┌────────────────────────────────┐
│  [User]──[Order]──[Payment]    │   내부 함수 호출
│  [Inventory]──[Notification]   │   네트워크 복잡성 없음
└────────────────────────────────┘

마이크로서비스:
┌──────┐    ┌──────┐    ┌─────────┐    ┌──────────┐
│ User │◀──▶│Order │◀──▶│ Payment │◀──▶│Inventory │
│  svc │    │ svc  │    │   svc   │    │   svc    │
└──┬───┘    └──┬───┘    └────┬────┘    └──────────┘
   │           │             │
   ▼           ▼             ▼
┌───────┐  ┌──────┐    ┌──────────┐
│Notify │  │ Cart │    │  Refund  │
│  svc  │  │ svc  │    │   svc    │
└───────┘  └──────┘    └──────────┘

모든 화살표에 필요한 것:
  - 서비스 디스커버리 (대상이 어디에 있는가?)
  - 로드밸런싱 (어떤 인스턴스로?)
  - 재시도 (실패하면?)
  - 타임아웃 (얼마나 기다릴 것인가?)
  - 서킷 브레이킹 (대상이 다운되면 호출 중지)
  - mTLS (암호화 및 인증)
  - 메트릭 (지연 시간, 오류율)
  - 트레이싱 (서비스 간 요청 추적)
```

### 서비스 메시가 제공하는 것

| 기능 | 메시 없이 | 메시 사용 시 |
|------|----------|------------|
| **로드밸런싱** | 클라이언트 측 라이브러리 (언어별) | 자동 (프록시 수준) |
| **재시도/타임아웃** | 각 서비스가 자체 구현 | 선언적으로 구성 |
| **서킷 브레이킹** | 언어별 라이브러리 (Hystrix 등) | 프록시 수준, 언어 무관 |
| **mTLS** | 각 서비스가 인증서를 관리 | 자동 인증서 순환 |
| **관찰 가능성** | 각 서비스를 수동으로 계측 | 자동 메트릭, 트레이스, 로그 |
| **트래픽 제어** | 애플리케이션 수준 라우팅 로직 | 선언적 트래픽 정책 |
| **접근 제어** | 서비스별 인증 미들웨어 | 중앙 집중식 인가 정책 |

---

## 2. 서비스 메시 아키텍처

### 사이드카 패턴

```
사이드카 없이:
┌─────────────────────────────┐
│  Application Container      │
│  ┌───────────────────────┐  │
│  │  비즈니스 로직         │  │
│  │  + 재시도 로직         │  │
│  │  + 서킷 브레이커       │  │
│  │  + TLS 관리           │  │
│  │  + 메트릭 수집         │  │
│  │  + 트레이싱           │  │
│  └───────────────────────┘  │
└─────────────────────────────┘

사이드카 사용:
┌──────────────────────────────────────────┐
│  Pod                                      │
│  ┌──────────────────┐  ┌──────────────┐  │
│  │  Application     │  │  Sidecar     │  │
│  │  Container       │◀▶│  Proxy       │  │
│  │                  │  │  (Envoy)     │  │
│  │  비즈니스 로직    │  │  - Retry     │  │
│  │  ONLY            │  │  - mTLS      │  │
│  │                  │  │  - Metrics   │  │
│  │                  │  │  - Tracing   │  │
│  └──────────────────┘  └──────┬───────┘  │
└───────────────────────────────┼──────────┘
                                │
                        모든 트래픽이
                        프록시를 통해 흐릅니다
```

### 컨트롤 플레인 vs 데이터 플레인

```
┌──────────────────────────────────────────────────────────┐
│                    Control Plane (istiod)                  │
│                                                           │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Pilot   │  │  Citadel     │  │  Galley            │  │
│  │ (트래픽   │  │ (인증서      │  │ (구성              │  │
│  │  구성)    │  │  인증 기관)  │  │  검증)             │  │
│  └──────────┘  └──────────────┘  └────────────────────┘  │
│       │              │                    │                │
│       └──────────────┼────────────────────┘                │
│                      │                                     │
│              프록시에 구성을 푸시                           │
└──────────────────────┼─────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Pod A       │ │  Pod B       │ │  Pod C       │
│ ┌────┐┌────┐│ │ ┌────┐┌────┐│ │ ┌────┐┌────┐│
│ │ App││Envy││ │ │ App││Envy││ │ │ App││Envy││
│ └────┘└────┘│ │ └────┘└────┘│ │ └────┘└────┘│
└──────────────┘ └──────────────┘ └──────────────┘
     Data Plane (Envoy 사이드카 프록시)
```

### Istio 구성 요소

| 구성 요소 | 역할 |
|----------|------|
| **istiod** | 통합 컨트롤 플레인 (Pilot, Citadel, Galley를 결합) |
| **Pilot** | 라우팅 규칙을 Envoy 구성으로 변환, 서비스 디스커버리 |
| **Citadel** | mTLS를 위한 인증서 인증 기관, 신원 관리 |
| **Galley** | 구성 검증 및 배포 |
| **Envoy** | 고성능 프록시 (데이터 플레인), 사이드카로 배포 |

---

## 3. Istio 설치

```bash
# Istio CLI 다운로드
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.21.0
export PATH=$PWD/bin:$PATH

# demo 프로필로 Istio 설치 (모든 컴포넌트 포함)
istioctl install --set profile=demo -y

# 프로필:
#   minimal  - istiod만 (ingress/egress 게이트웨이 없음)
#   default  - istiod + ingress 게이트웨이
#   demo     - istiod + ingress + egress + 애드온 (학습용)
#   production - 프로덕션을 위한 강화된 기본값

# 설치 확인
istioctl verify-install

# 네임스페이스에 대한 자동 사이드카 주입 활성화
kubectl label namespace default istio-injection=enabled

# 사이드카 주입 확인
kubectl get namespace -L istio-injection

# 테스트 애플리케이션 배포 (Istio의 Bookinfo 샘플)
kubectl apply -f samples/bookinfo/platform/kube/bookinfo.yaml

# 파드에 2개의 컨테이너가 있는지 확인 (앱 + envoy 사이드카)
kubectl get pods
# NAME                             READY   STATUS    RESTARTS   AGE
# productpage-v1-abc123            2/2     Running   0          1m
#                                  ^^^
#                                  2 containers = 사이드카가 주입됨
```

---

## 4. 트래픽 관리

### VirtualService

VirtualService는 요청이 서비스로 라우팅되는 방식을 정의합니다.

```yaml
# 모든 트래픽을 reviews 서비스의 v1으로 라우팅
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
    - reviews                              # Kubernetes service name
  http:
    - route:
        - destination:
            host: reviews
            subset: v1                     # Route to subset defined in DestinationRule
```

### DestinationRule

DestinationRule은 서브셋(버전)과 트래픽 정책을 정의합니다.

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: DEFAULT
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    loadBalancer:
      simple: ROUND_ROBIN

  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
    - name: v3
      labels:
        version: v3
```

### 카나리 배포 (트래픽 분할)

```yaml
# 트래픽의 90%를 v1으로, 10%를 v2(카나리)로 전송
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
    - reviews
  http:
    - route:
        - destination:
            host: reviews
            subset: v1
          weight: 90                       # 90% to v1
        - destination:
            host: reviews
            subset: v2
          weight: 10                       # 10% to v2 (canary)
```

### 점진적 카나리 롤아웃

```yaml
# Phase 1: 5% 카나리
# Phase 2: 25% 카나리 (메트릭이 양호하면)
# Phase 3: 50% 카나리
# Phase 4: 100% (전체 롤아웃)

# Phase 1:
http:
  - route:
      - destination: { host: reviews, subset: v1 }
        weight: 95
      - destination: { host: reviews, subset: v2 }
        weight: 5

# Phase 4 (전체 롤아웃):
http:
  - route:
      - destination: { host: reviews, subset: v2 }
        weight: 100
```

### 헤더 기반 라우팅

```yaml
# 내부 테스터를 v2로, 나머지는 v1으로 라우팅
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
    - reviews
  http:
    # 특정 헤더 매칭 -- 내부 테스터는 v2를 봅니다
    - match:
        - headers:
            x-test-user:
              exact: "true"
      route:
        - destination:
            host: reviews
            subset: v2

    # 기본 라우트 -- 나머지는 v1
    - route:
        - destination:
            host: reviews
            subset: v1
```

### 재시도와 타임아웃

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
    - reviews
  http:
    - route:
        - destination:
            host: reviews
            subset: v1
      timeout: 5s                          # Request timeout

      retries:
        attempts: 3                        # Retry up to 3 times
        perTryTimeout: 2s                  # Timeout per retry attempt
        retryOn: "5xx,reset,connect-failure,retriable-4xx"
```

### 장애 주입 (카오스 테스팅)

```yaml
# 요청의 10%에 5초 지연을 주입 (복원력 테스트)
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
    - reviews
  http:
    - fault:
        delay:
          percentage:
            value: 10.0                    # 10% of requests
          fixedDelay: 5s                   # 5 second delay

        abort:
          percentage:
            value: 5.0                     # 5% of requests
          httpStatus: 503                  # Return 503 error

      route:
        - destination:
            host: reviews
            subset: v1
```

### 서킷 브레이킹

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100                # Max TCP connections
      http:
        http1MaxPendingRequests: 100       # Max pending HTTP/1.1 requests
        http2MaxRequests: 1000             # Max HTTP/2 requests
        maxRequestsPerConnection: 10       # Max requests per connection

    outlierDetection:
      consecutive5xxErrors: 5              # Eject after 5 consecutive 5xx errors
      interval: 30s                        # Check every 30 seconds
      baseEjectionTime: 30s               # Eject for 30 seconds
      maxEjectionPercent: 50              # Never eject more than 50% of endpoints
```

---

## 5. 상호 TLS (mTLS)

mTLS는 서비스 간 호출에서 클라이언트와 서버 모두가 인증서를 사용하여 상대방의 신원을 검증하도록 보장합니다.

### Istio에서 mTLS 작동 방식

```
┌──────────────────────────────────────────────────────────┐
│                     mTLS in Istio                         │
│                                                           │
│  Service A                        Service B               │
│  ┌──────┐  ┌──────┐    mTLS    ┌──────┐  ┌──────┐       │
│  │ App  │──│Envoy │◀══════════▶│Envoy │──│ App  │       │
│  │      │  │proxy │  encrypted │proxy │  │      │       │
│  └──────┘  └──────┘  + authed  └──────┘  └──────┘       │
│                │                    │                      │
│                ▼                    ▼                      │
│          Certificate           Certificate                │
│          from istiod           from istiod                │
│                                                           │
│  istiod (Citadel):                                       │
│    - 모든 사이드카에 인증서 발급                           │
│    - 인증서 자동 순환 (기본 24시간)                        │
│    - 애플리케이션 코드 변경 불필요                         │
└──────────────────────────────────────────────────────────┘
```

### mTLS 모드

```yaml
# Strict mTLS -- 모든 트래픽이 암호화되어야 합니다
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production               # Apply to entire namespace
spec:
  mtls:
    mode: STRICT                       # Only accept mTLS connections

# Permissive mTLS -- 평문과 mTLS 모두 수용 (마이그레이션 모드)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: PERMISSIVE                   # Accept both (useful during migration)
```

### 서비스별 mTLS 재정의

```yaml
# 특정 서비스에 대해 mTLS 비활성화 (예: 레거시 서비스)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: legacy-exception
  namespace: production
spec:
  selector:
    matchLabels:
      app: legacy-service              # Only apply to this service
  mtls:
    mode: DISABLE
```

### 인가 정책(Authorization Policy)

```yaml
# 특정 서비스만 결제 서비스를 호출할 수 있도록 허용
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: payment-policy
  namespace: production
spec:
  selector:
    matchLabels:
      app: payment-service

  rules:
    - from:
        - source:
            principals:
              - "cluster.local/ns/production/sa/order-service"
              - "cluster.local/ns/production/sa/refund-service"
      to:
        - operation:
            methods: ["POST"]
            paths: ["/api/v1/charge", "/api/v1/refund"]

    # 나머지는 모두 거부 (규칙이 있을 때 암묵적 거부)
```

---

## 6. 관찰 가능성(Observability)

Istio의 사이드카 프록시는 모든 서비스 간 트래픽에 대해 메트릭, 트레이스, 접근 로그를 자동으로 수집합니다.

### 메트릭 (Prometheus + Grafana)

```bash
# Istio가 자동으로 메트릭을 생성합니다:
#   - istio_requests_total (요청 수)
#   - istio_request_duration_milliseconds (지연 시간)
#   - istio_request_bytes / istio_response_bytes (페이로드 크기)

# Prometheus와 Grafana 애드온 설치
kubectl apply -f samples/addons/prometheus.yaml
kubectl apply -f samples/addons/grafana.yaml

# Grafana 대시보드 접근
istioctl dashboard grafana
# http://localhost:3000에서 사전 구축된 Istio 대시보드를 엽니다:
#   - Mesh Dashboard (전체 뷰)
#   - Service Dashboard (서비스별 메트릭)
#   - Workload Dashboard (워크로드별 메트릭)
```

### 주요 모니터링 메트릭

```
요청 속도:
  rate(istio_requests_total{reporter="destination"}[5m])

오류율:
  rate(istio_requests_total{reporter="destination",response_code=~"5.."}[5m])
  /
  rate(istio_requests_total{reporter="destination"}[5m])

지연 시간 (P99):
  histogram_quantile(0.99,
    rate(istio_request_duration_milliseconds_bucket{reporter="destination"}[5m])
  )

성공률:
  sum(rate(istio_requests_total{reporter="destination",response_code!~"5.."}[5m]))
  /
  sum(rate(istio_requests_total{reporter="destination"}[5m]))
```

### 분산 트레이싱 (Jaeger)

```bash
# Jaeger 설치
kubectl apply -f samples/addons/jaeger.yaml

# Jaeger UI 접근
istioctl dashboard jaeger
# http://localhost:16686을 엽니다

# 트레이싱은 서비스 간 요청의 전체 경로를 보여줍니다:
#
# 요청 흐름:
# [Client] → [Ingress] → [ProductPage] → [Reviews] → [Ratings]
#                                       → [Details]
#
# Jaeger 트레이스:
# ├── ingress-gateway (2ms)
# │   └── productpage (45ms)
# │       ├── reviews (30ms)
# │       │   └── ratings (8ms)
# │       └── details (12ms)
```

### 서비스 그래프 (Kiali)

```bash
# Kiali 설치 (서비스 메시 관찰 가능성 대시보드)
kubectl apply -f samples/addons/kiali.yaml

# Kiali 대시보드 접근
istioctl dashboard kiali
# http://localhost:20001을 엽니다

# Kiali가 제공하는 기능:
#   - 실시간 서비스 토폴로지 그래프
#   - 트래픽 흐름 시각화
#   - 서비스별 상태 표시기
#   - 구성 검증
#   - mTLS 상태 시각화
```

### 접근 로깅

```yaml
# 모든 사이드카에 대한 접근 로깅 활성화
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: mesh-default
  namespace: istio-system
spec:
  accessLogging:
    - providers:
        - name: envoy
      filter:
        expression: "response.code >= 400"  # Only log errors
```

---

## 7. Ingress Gateway

Istio Ingress Gateway는 메시 외부에서 들어오는 트래픽을 관리합니다.

```yaml
# Gateway -- 진입점을 정의합니다
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: app-gateway
spec:
  selector:
    istio: ingressgateway               # Use Istio's ingress gateway
  servers:
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts:
        - "app.example.com"
      tls:
        httpsRedirect: true             # Redirect HTTP to HTTPS

    - port:
        number: 443
        name: https
        protocol: HTTPS
      hosts:
        - "app.example.com"
      tls:
        mode: SIMPLE
        credentialName: app-tls-cert    # Kubernetes TLS Secret
```

```yaml
# Gateway에 바인딩된 VirtualService
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: app-routing
spec:
  hosts:
    - "app.example.com"
  gateways:
    - app-gateway                       # Bind to the Gateway above
  http:
    - match:
        - uri:
            prefix: /api/
      route:
        - destination:
            host: api-service
            port:
              number: 80

    - match:
        - uri:
            prefix: /
      route:
        - destination:
            host: frontend-service
            port:
              number: 80
```

---

## 8. 서비스 메시 대안

| 특성 | Istio | Linkerd | Consul Connect | Cilium |
|------|-------|---------|----------------|--------|
| **프록시** | Envoy | linkerd2-proxy (Rust) | Envoy (선택적) | eBPF (사이드카 없음) |
| **복잡도** | 높음 | 낮음 | 중간 | 중간 |
| **성능** | 양호 | 우수 (경량) | 양호 | 우수 (커널 수준) |
| **mTLS** | 예 | 예 (기본 활성화) | 예 | 예 |
| **트래픽 관리** | 매우 풍부 | 기본적 | 보통 | 보통 |
| **멀티 클러스터** | 예 | 예 | 예 (네이티브) | 예 |
| **적합한 경우** | 풍부한 기능이 필요할 때 | 단순성 | HashiCorp 스택 | 성능 |

### 사용 시기 (및 사용하지 않을 시기)

```
서비스 메시를 사용해야 할 때:
  ✓ 마이크로서비스가 10개 이상인 경우
  ✓ 서비스 간 mTLS가 필요한 경우
  ✓ 세밀한 트래픽 제어가 필요한 경우 (카나리, A/B)
  ✓ 서비스 전반에 걸친 통합 관찰 가능성이 필요한 경우
  ✓ 규정 준수를 위해 암호화된 서비스 간 트래픽이 필요한 경우

서비스 메시를 건너뛰어야 할 때:
  ✗ 서비스가 5개 미만인 경우 (오버헤드가 정당화되지 않음)
  ✗ 팀이 Kubernetes에 익숙하지 않은 경우 (나중에 메시 추가)
  ✗ mTLS나 고급 트래픽 라우팅이 필요하지 않은 경우
  ✗ 성능 오버헤드가 허용되지 않는 경우 (Cilium/eBPF 고려)
  ✗ 모놀리스를 실행하고 있는 경우
```

---

## 9. Istio 문제 해결

```bash
# Istio 컴포넌트 상태 확인
istioctl proxy-status

# 메시 구성에 대한 문제 분석
istioctl analyze
istioctl analyze -n production

# 특정 파드의 사이드카 구성 확인
istioctl proxy-config routes deploy/api-server -n production
istioctl proxy-config clusters deploy/api-server -n production
istioctl proxy-config endpoints deploy/api-server -n production

# 서비스 간 mTLS 활성화 여부 확인
istioctl authn tls-check api-server.production.svc.cluster.local

# 특정 프록시 디버깅
istioctl dashboard envoy deploy/api-server -n production

# Istio 구성 조회 (VirtualService, DestinationRule 등)
kubectl get virtualservices,destinationrules,gateways -n production

# Istio 로그 확인
kubectl logs -n istio-system deploy/istiod
```

---

## 연습 문제

### 연습 문제 1: Istio를 이용한 카나리 배포

웹 애플리케이션에 대한 카나리 배포를 구현하십시오:
1. 서로 다른 Deployment 레이블을 가진 두 버전의 애플리케이션(v1과 v2)을 배포합니다
2. v1과 v2에 대한 서브셋이 있는 DestinationRule을 생성합니다
3. 트래픽의 90%를 v1으로, 10%를 v2로 전송하는 VirtualService를 생성합니다
4. 100개의 요청을 전송하고 응답을 세어 트래픽 분배를 확인합니다
5. v2 트래픽을 25%, 50%, 최종적으로 100%로 점진적으로 증가시킵니다
6. 각 단계에서 Kiali 또는 Grafana에서 오류율을 모니터링합니다

### 연습 문제 2: 헤더 기반 트래픽 라우팅

헤더 기반 트래픽 라우팅을 설정하십시오:
1. 서비스의 v1과 v2를 배포합니다
2. `x-version: v2` 헤더가 있는 요청을 v2 서브셋으로 라우팅하는 VirtualService를 생성합니다
3. 나머지 모든 트래픽은 v1으로 이동합니다
4. `curl -H "x-version: v2" http://<service>/`로 테스트합니다
5. 두 번째 규칙을 추가합니다: "Mobile"을 포함하는 사용자 에이전트의 요청을 모바일 최적화 버전으로 라우팅

### 연습 문제 3: 복원력 테스트

Istio 장애 주입을 사용하여 서비스 복원력을 테스트하십시오:
1. 서비스 체인을 배포합니다: frontend -> backend -> database
2. backend 요청의 50%에 3초 지연을 주입합니다
3. frontend가 지연을 우아하게 처리하는지 (또는 타임아웃하는지) 확인합니다
4. backend 요청의 20%에 503 오류를 주입합니다
5. 재시도(3회 시도)를 구성하고 유효 오류율이 감소하는지 확인합니다
6. 서킷 브레이킹을 추가합니다: 3번 연속 5xx 오류 후 backend를 축출

### 연습 문제 4: mTLS 구성

서비스 간 통신을 보호하십시오:
1. 사이드카 주입이 활성화된 Istio를 설치합니다
2. 서로 통신하는 두 서비스를 배포합니다
3. 기본적으로 트래픽이 암호화되지 않는 것을 확인합니다 (PERMISSIVE 모드)
4. 네임스페이스에 대해 STRICT mTLS로 전환합니다
5. 사이드카가 없는 파드가 메시 서비스와 더 이상 통신할 수 없는 것을 확인합니다
6. `istioctl`과 Kiali를 사용하여 mTLS 상태를 확인합니다

### 연습 문제 5: 관찰 가능성 설정

마이크로서비스 애플리케이션에 대한 전체 관찰 가능성을 설정하십시오:
1. Prometheus, Grafana, Jaeger, Kiali 애드온과 함께 Istio를 배포합니다
2. Bookinfo 샘플 애플리케이션을 배포합니다
3. 부하 생성기를 사용하여 트래픽을 생성합니다
4. Grafana에서: productpage 서비스의 P99 지연 시간을 찾습니다
5. Jaeger에서: 모든 서비스를 통한 단일 요청을 추적하고 가장 느린 구간을 식별합니다
6. Kiali에서: 서비스 그래프를 보고 오류율이 1%를 초과하는 서비스를 식별합니다

---

**이전**: [컨테이너 오케스트레이션 운영](./08_Container_Orchestration_Operations.md) | [개요](00_Overview.md)

**License**: CC BY-NC 4.0
