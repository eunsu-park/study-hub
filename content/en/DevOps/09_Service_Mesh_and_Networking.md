# Lesson 9: Service Mesh and Networking

**Previous**: [Container Orchestration Operations](./08_Container_Orchestration_Operations.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the service mesh pattern and why it emerged to handle microservice networking complexity
2. Describe Istio's architecture including the control plane (istiod) and data plane (Envoy sidecars)
3. Configure traffic management policies including canary deployments, traffic splitting, retries, and circuit breaking
4. Implement mutual TLS (mTLS) for zero-trust service-to-service communication
5. Use Istio's observability features to gain visibility into service traffic, latency, and error rates

---

As applications evolve from monoliths into dozens or hundreds of microservices, the networking between services becomes a critical concern. Each service needs load balancing, retries, timeouts, circuit breaking, mutual authentication, and observability -- and implementing these in every service's application code creates enormous duplication and inconsistency. A service mesh solves this by moving networking logic out of the application and into a dedicated infrastructure layer. This lesson covers service mesh concepts with a deep dive into Istio, the most widely adopted service mesh.

> **Analogy -- Postal System:** Without a service mesh, every microservice is like a person who has to personally deliver every letter: find the address, choose the route, handle failures, track delivery. A service mesh is like a postal system: each service drops messages in a mailbox (the sidecar proxy), and the postal infrastructure handles routing, delivery confirmation, retry on failure, and tracking -- without the sender needing to know the details.

## 1. Why Service Mesh?

### The Microservice Networking Problem

```
Monolith:
┌────────────────────────────────┐
│  [User]──[Order]──[Payment]    │   Internal function calls
│  [Inventory]──[Notification]   │   No network complexity
└────────────────────────────────┘

Microservices:
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

Every arrow needs:
  - Service discovery (where is the target?)
  - Load balancing (which instance?)
  - Retries (what if it fails?)
  - Timeouts (how long to wait?)
  - Circuit breaking (stop calling if target is down)
  - mTLS (encrypt and authenticate)
  - Metrics (latency, error rate)
  - Tracing (follow a request across services)
```

### What a Service Mesh Provides

| Capability | Without Mesh | With Mesh |
|-----------|-------------|-----------|
| **Load balancing** | Client-side libraries (per language) | Automatic (proxy-level) |
| **Retries/timeouts** | Each service implements its own | Configured declaratively |
| **Circuit breaking** | Library per language (Hystrix, etc.) | Proxy-level, language-agnostic |
| **mTLS** | Each service manages certificates | Automatic certificate rotation |
| **Observability** | Instrument each service manually | Automatic metrics, traces, logs |
| **Traffic control** | Application-level routing logic | Declarative traffic policies |
| **Access control** | Per-service auth middleware | Centralized authorization policies |

---

## 2. Service Mesh Architecture

### Sidecar Pattern

```
Without sidecar:
┌─────────────────────────────┐
│  Application Container      │
│  ┌───────────────────────┐  │
│  │  Business Logic       │  │
│  │  + Retry Logic        │  │
│  │  + Circuit Breaker    │  │
│  │  + TLS Management     │  │
│  │  + Metrics Collection │  │
│  │  + Tracing            │  │
│  └───────────────────────┘  │
└─────────────────────────────┘

With sidecar:
┌──────────────────────────────────────────┐
│  Pod                                      │
│  ┌──────────────────┐  ┌──────────────┐  │
│  │  Application     │  │  Sidecar     │  │
│  │  Container       │◀▶│  Proxy       │  │
│  │                  │  │  (Envoy)     │  │
│  │  Business Logic  │  │  - Retry     │  │
│  │  ONLY            │  │  - mTLS      │  │
│  │                  │  │  - Metrics   │  │
│  │                  │  │  - Tracing   │  │
│  └──────────────────┘  └──────┬───────┘  │
└───────────────────────────────┼──────────┘
                                │
                        All traffic flows
                        through the proxy
```

### Control Plane vs Data Plane

```
┌──────────────────────────────────────────────────────────┐
│                    Control Plane (istiod)                  │
│                                                           │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Pilot   │  │  Citadel     │  │  Galley            │  │
│  │ (traffic │  │ (certificate │  │ (configuration     │  │
│  │  config) │  │  authority)  │  │  validation)       │  │
│  └──────────┘  └──────────────┘  └────────────────────┘  │
│       │              │                    │                │
│       └──────────────┼────────────────────┘                │
│                      │                                     │
│              Push configuration to proxies                 │
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
     Data Plane (Envoy sidecar proxies)
```

### Istio Components

| Component | Role |
|-----------|------|
| **istiod** | Unified control plane (combines Pilot, Citadel, Galley) |
| **Pilot** | Converts routing rules to Envoy configuration, service discovery |
| **Citadel** | Certificate authority for mTLS, identity management |
| **Galley** | Configuration validation and distribution |
| **Envoy** | High-performance proxy (data plane), deployed as sidecar |

---

## 3. Installing Istio

```bash
# Download Istio CLI
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.21.0
export PATH=$PWD/bin:$PATH

# Install Istio with the demo profile (includes all components)
istioctl install --set profile=demo -y

# Profiles:
#   minimal  - Only istiod (no ingress/egress gateway)
#   default  - istiod + ingress gateway
#   demo     - istiod + ingress + egress + add-ons (for learning)
#   production - Hardened defaults for production

# Verify installation
istioctl verify-install

# Enable automatic sidecar injection for a namespace
kubectl label namespace default istio-injection=enabled

# Verify sidecar injection
kubectl get namespace -L istio-injection

# Deploy a test application (Istio's Bookinfo sample)
kubectl apply -f samples/bookinfo/platform/kube/bookinfo.yaml

# Verify pods have 2 containers (app + envoy sidecar)
kubectl get pods
# NAME                             READY   STATUS    RESTARTS   AGE
# productpage-v1-abc123            2/2     Running   0          1m
#                                  ^^^
#                                  2 containers = sidecar injected
```

---

## 4. Traffic Management

### VirtualService

VirtualService defines how requests are routed to services.

```yaml
# Route all traffic to v1 of the reviews service
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

DestinationRule defines subsets (versions) and traffic policies.

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

### Canary Deployment (Traffic Splitting)

```yaml
# Send 90% of traffic to v1, 10% to v2 (canary)
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

### Progressive Canary Rollout

```yaml
# Phase 1: 5% canary
# Phase 2: 25% canary (if metrics look good)
# Phase 3: 50% canary
# Phase 4: 100% (full rollout)

# Phase 1:
http:
  - route:
      - destination: { host: reviews, subset: v1 }
        weight: 95
      - destination: { host: reviews, subset: v2 }
        weight: 5

# Phase 4 (full rollout):
http:
  - route:
      - destination: { host: reviews, subset: v2 }
        weight: 100
```

### Header-Based Routing

```yaml
# Route internal testers to v2, everyone else to v1
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
    - reviews
  http:
    # Match specific header -- internal testers see v2
    - match:
        - headers:
            x-test-user:
              exact: "true"
      route:
        - destination:
            host: reviews
            subset: v2

    # Default route -- everyone else gets v1
    - route:
        - destination:
            host: reviews
            subset: v1
```

### Retries and Timeouts

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

### Fault Injection (Chaos Testing)

```yaml
# Inject a 5-second delay into 10% of requests (test resilience)
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

### Circuit Breaking

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

## 5. Mutual TLS (mTLS)

mTLS ensures that both the client and server in a service-to-service call verify each other's identity using certificates.

### How mTLS Works in Istio

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
│    - Issues certificates to every sidecar                │
│    - Rotates certificates automatically (24h default)     │
│    - No application code changes needed                  │
└──────────────────────────────────────────────────────────┘
```

### mTLS Modes

```yaml
# Strict mTLS -- ALL traffic must be encrypted
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production               # Apply to entire namespace
spec:
  mtls:
    mode: STRICT                       # Only accept mTLS connections

# Permissive mTLS -- accept both plain text and mTLS (migration mode)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: PERMISSIVE                   # Accept both (useful during migration)
```

### Per-Service mTLS Override

```yaml
# Disable mTLS for a specific service (e.g., legacy service)
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

### Authorization Policy

```yaml
# Only allow specific services to call the payment service
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

    # Deny everything else (implicit deny when rules are present)
```

---

## 6. Observability

Istio's sidecar proxies automatically collect metrics, traces, and access logs for all service-to-service traffic.

### Metrics (Prometheus + Grafana)

```bash
# Istio generates metrics automatically:
#   - istio_requests_total (request count)
#   - istio_request_duration_milliseconds (latency)
#   - istio_request_bytes / istio_response_bytes (payload size)

# Install Prometheus and Grafana add-ons
kubectl apply -f samples/addons/prometheus.yaml
kubectl apply -f samples/addons/grafana.yaml

# Access Grafana dashboard
istioctl dashboard grafana
# Opens http://localhost:3000 with pre-built Istio dashboards:
#   - Mesh Dashboard (global view)
#   - Service Dashboard (per-service metrics)
#   - Workload Dashboard (per-workload metrics)
```

### Key Metrics to Monitor

```
Request Rate:
  rate(istio_requests_total{reporter="destination"}[5m])

Error Rate:
  rate(istio_requests_total{reporter="destination",response_code=~"5.."}[5m])
  /
  rate(istio_requests_total{reporter="destination"}[5m])

Latency (P99):
  histogram_quantile(0.99,
    rate(istio_request_duration_milliseconds_bucket{reporter="destination"}[5m])
  )

Success Rate:
  sum(rate(istio_requests_total{reporter="destination",response_code!~"5.."}[5m]))
  /
  sum(rate(istio_requests_total{reporter="destination"}[5m]))
```

### Distributed Tracing (Jaeger)

```bash
# Install Jaeger
kubectl apply -f samples/addons/jaeger.yaml

# Access Jaeger UI
istioctl dashboard jaeger
# Opens http://localhost:16686

# Tracing shows the full path of a request across services:
#
# Request flow:
# [Client] → [Ingress] → [ProductPage] → [Reviews] → [Ratings]
#                                       → [Details]
#
# Jaeger trace:
# ├── ingress-gateway (2ms)
# │   └── productpage (45ms)
# │       ├── reviews (30ms)
# │       │   └── ratings (8ms)
# │       └── details (12ms)
```

### Service Graph (Kiali)

```bash
# Install Kiali (service mesh observability dashboard)
kubectl apply -f samples/addons/kiali.yaml

# Access Kiali dashboard
istioctl dashboard kiali
# Opens http://localhost:20001

# Kiali provides:
#   - Real-time service topology graph
#   - Traffic flow visualization
#   - Health indicators per service
#   - Configuration validation
#   - mTLS status visualization
```

### Access Logging

```yaml
# Enable access logging for all sidecars
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

The Istio Ingress Gateway manages incoming traffic from outside the mesh.

```yaml
# Gateway -- defines the entry point
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
# VirtualService bound to the Gateway
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

## 8. Service Mesh Alternatives

| Feature | Istio | Linkerd | Consul Connect | Cilium |
|---------|-------|---------|----------------|--------|
| **Proxy** | Envoy | linkerd2-proxy (Rust) | Envoy (optional) | eBPF (no sidecar) |
| **Complexity** | High | Low | Medium | Medium |
| **Performance** | Good | Excellent (lightweight) | Good | Excellent (kernel-level) |
| **mTLS** | Yes | Yes (default on) | Yes | Yes |
| **Traffic mgmt** | Very rich | Basic | Moderate | Moderate |
| **Multi-cluster** | Yes | Yes | Yes (native) | Yes |
| **Best for** | Feature-rich needs | Simplicity | HashiCorp stack | Performance |

### When to Use (and When Not To)

```
USE a service mesh when:
  ✓ You have 10+ microservices
  ✓ You need mTLS between services
  ✓ You need fine-grained traffic control (canary, A/B)
  ✓ You need unified observability across services
  ✓ Compliance requires encrypted service-to-service traffic

SKIP a service mesh when:
  ✗ You have < 5 services (overhead is not justified)
  ✗ Your team is new to Kubernetes (add mesh later)
  ✗ You don't need mTLS or advanced traffic routing
  ✗ Performance overhead is unacceptable (consider Cilium/eBPF)
  ✗ You are running a monolith
```

---

## 9. Troubleshooting Istio

```bash
# Check Istio component status
istioctl proxy-status

# Analyze mesh configuration for issues
istioctl analyze
istioctl analyze -n production

# Check sidecar configuration for a specific pod
istioctl proxy-config routes deploy/api-server -n production
istioctl proxy-config clusters deploy/api-server -n production
istioctl proxy-config endpoints deploy/api-server -n production

# Check if mTLS is active between services
istioctl authn tls-check api-server.production.svc.cluster.local

# Debug a specific proxy
istioctl dashboard envoy deploy/api-server -n production

# View Istio configuration (VirtualServices, DestinationRules, etc.)
kubectl get virtualservices,destinationrules,gateways -n production

# Check Istio logs
kubectl logs -n istio-system deploy/istiod
```

---

## Exercises

### Exercise 1: Canary Deployment with Istio

Implement a canary deployment for a web application:
1. Deploy two versions of an application (v1 and v2) with different Deployment labels
2. Create a DestinationRule with subsets for v1 and v2
3. Create a VirtualService that sends 90% of traffic to v1 and 10% to v2
4. Verify traffic distribution by sending 100 requests and counting responses
5. Gradually increase v2 traffic to 25%, 50%, and finally 100%
6. Monitor error rates in Kiali or Grafana during each phase

### Exercise 2: Traffic Routing by Headers

Set up header-based traffic routing:
1. Deploy v1 and v2 of a service
2. Create a VirtualService that routes requests with header `x-version: v2` to the v2 subset
3. All other traffic goes to v1
4. Test with `curl -H "x-version: v2" http://<service>/`
5. Add a second rule: route requests from user agent containing "Mobile" to a mobile-optimized version

### Exercise 3: Resilience Testing

Test service resilience using Istio fault injection:
1. Deploy a chain of services: frontend -> backend -> database
2. Inject a 3-second delay into 50% of backend requests
3. Verify that the frontend handles the delay gracefully (or times out)
4. Inject 503 errors into 20% of backend requests
5. Configure retries (3 attempts) and verify that the effective error rate drops
6. Add circuit breaking: eject the backend after 3 consecutive 5xx errors

### Exercise 4: mTLS Configuration

Secure service-to-service communication:
1. Install Istio with sidecar injection enabled
2. Deploy two services that communicate with each other
3. Verify that traffic is unencrypted by default (PERMISSIVE mode)
4. Switch to STRICT mTLS for the namespace
5. Verify that a pod without a sidecar can no longer communicate with mesh services
6. Check mTLS status using `istioctl` and Kiali

### Exercise 5: Observability Setup

Set up full observability for a microservice application:
1. Deploy Istio with Prometheus, Grafana, Jaeger, and Kiali add-ons
2. Deploy the Bookinfo sample application
3. Generate traffic using a load generator
4. In Grafana: find the P99 latency for the productpage service
5. In Jaeger: trace a single request through all services and identify the slowest hop
6. In Kiali: view the service graph and identify any services with error rates above 1%

---

**Previous**: [Container Orchestration Operations](./08_Container_Orchestration_Operations.md) | [Overview](00_Overview.md)

**License**: CC BY-NC 4.0
