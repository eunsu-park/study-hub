# Microservices Patterns

**Previous**: [Microservices Fundamentals](./13_Microservices_Basics.md) | **Next**: [Distributed Systems Concepts](./15_Distributed_Systems_Concepts.md)

Difficulty: ⭐⭐⭐⭐

## Learning Objectives

1. Explain client-side and server-side service discovery mechanisms and compare registry-based approaches (Consul, Eureka) with DNS-based solutions
2. Implement the Circuit Breaker pattern to prevent cascading failures across dependent services
3. Apply the Bulkhead pattern to isolate failures and protect critical resources from being exhausted by a single component
4. Describe the architecture of a service mesh (data plane vs control plane) and the role of sidecar proxies
5. Design a distributed tracing strategy using trace context propagation, span hierarchies, and sampling policies
6. Evaluate when to apply each pattern based on system maturity, operational complexity, and failure modes

---

Building microservices is the easy part -- keeping hundreds of them running reliably in production is where the real engineering challenge begins. A single slow downstream service can cascade into a system-wide outage if you have no circuit breakers, and debugging a request that touches twelve services is nearly impossible without distributed tracing. The patterns in this lesson are the operational backbone that separates a resilient microservices deployment from a fragile one.

## Table of Contents

1. [Service Discovery](#1-service-discovery)
2. [Circuit Breaker](#2-circuit-breaker)
3. [Bulkhead Pattern](#3-bulkhead-pattern)
4. [Service Mesh](#4-service-mesh)
5. [Distributed Tracing](#5-distributed-tracing)
6. [Other Important Patterns](#6-other-important-patterns)
7. [Practice Problems](#7-practice-problems)

---

## 1. Service Discovery

### Why Is Service Discovery Needed?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     The Need for Service Discovery                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Traditional Approach (Hardcoding):                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Order Service                                                   │   │
│  │  config:                                                         │   │
│  │    user_service: http://10.0.1.5:8080                           │   │
│  │    product_service: http://10.0.1.10:8080                       │   │
│  │    payment_service: http://10.0.1.15:8080                       │   │
│  │                                                                  │   │
│  │  Problems:                                                       │   │
│  │  - Configuration changes required when IP changes                │   │
│  │  - Manual updates needed when scaling out                        │   │
│  │  - Cannot automatically exclude failed instances                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Problems in Dynamic Environments:                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Time 0:  User Service @ 10.0.1.5                               │   │
│  │  Time 1:  Scale out → 10.0.1.5, 10.0.1.6, 10.0.1.7             │   │
│  │  Time 2:  10.0.1.5 failure → 10.0.1.6, 10.0.1.7                │   │
│  │  Time 3:  New deployment → 10.0.1.8, 10.0.1.9                  │   │
│  │                                                                  │   │
│  │  → IPs keep changing! How do we track them?                     │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Client-Side Discovery

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Client-Side Discovery                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │           ┌────────────────────────────────┐                    │   │
│  │           │      Service Registry          │                    │   │
│  │           │   (Eureka, Consul, etcd)       │                    │   │
│  │           │                                │                    │   │
│  │           │  user-service:                 │                    │   │
│  │           │    - 10.0.1.5:8080            │                    │   │
│  │           │    - 10.0.1.6:8080            │                    │   │
│  │           │    - 10.0.1.7:8080            │                    │   │
│  │           │                                │                    │   │
│  │           └─────────────┬──────────────────┘                    │   │
│  │                         │                                        │   │
│  │    ┌─────1.Query────────┘                                       │   │
│  │    │    2.Return instances                                      │   │
│  │    ▼                                                             │   │
│  │  ┌───────────────┐      3.Direct call      ┌───────────────┐    │   │
│  │  │ Order Service │─────────────────────────│ User Service  │    │   │
│  │  │  (Client)     │   (Load Balancing)      │ (10.0.1.5)    │    │   │
│  │  │  + LB Logic   │─────────────────────────│ (10.0.1.6)    │    │   │
│  │  └───────────────┘                         │ (10.0.1.7)    │    │   │
│  │                                            └───────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  How It Works:                                                         │
│  1. Client queries the Registry for service instances                  │
│  2. Client performs load balancing (Round Robin, Random, etc.)         │
│  3. Direct instance call                                               │
│                                                                         │
│  Example: Netflix Eureka + Ribbon                                      │
│                                                                         │
│  Pros:                                                   Cons:         │
│  - Simple infrastructure                                - Complex client│
│  - Registry load distributed                            - Per-language  │
│                                                          implementation│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Server-Side Discovery

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Server-Side Discovery                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │           ┌────────────────────────────────┐                    │   │
│  │           │      Service Registry          │                    │   │
│  │           └─────────────┬──────────────────┘                    │   │
│  │                         │                                        │   │
│  │                         │ Instance info sync                     │   │
│  │                         ▼                                        │   │
│  │  ┌───────────────┐  ┌───────────────────┐   ┌───────────────┐   │   │
│  │  │ Order Service │─►│   Load Balancer   │──►│ User Service  │   │   │
│  │  │  (Client)     │  │  / API Gateway    │   │ (10.0.1.5)    │   │   │
│  │  │  Simple call  │  │                   │   │ (10.0.1.6)    │   │   │
│  │  └───────────────┘  │  - Routing        │   │ (10.0.1.7)    │   │   │
│  │                     │  - Load balancing │   └───────────────┘   │   │
│  │  GET /user-service  │  - Health check   │                       │   │
│  │  /users/123         └───────────────────┘                       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  How It Works:                                                         │
│  1. Client requests to LB/Gateway (using service name)                 │
│  2. LB queries instances from Registry                                 │
│  3. LB calls instance after load balancing                             │
│                                                                         │
│  Examples: AWS ELB + Route 53, Kubernetes Service, Nginx + Consul      │
│                                                                         │
│  Pros:                                        Cons:                    │
│  - Simplified client                          - LB is SPOF             │
│  - Language independent                       - Additional hop (latency)│
│  - Centralized management                     - LB operational cost    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Registration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Service Registration Patterns                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Self-Registration:                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  User Service                          Service Registry          │   │
│  │       │                                      │                   │   │
│  │  Start│─────── Register(ip, port) ─────────►│                   │   │
│  │       │                                      │                   │   │
│  │ Periodic──────── Heartbeat ─────────────────►│                   │   │
│  │       │                                      │                   │   │
│  │  Stop │─────── Deregister ─────────────────►│                   │   │
│  │       │                                      │                   │   │
│  │  Examples: Eureka Client, Consul Agent                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Third-Party Registration (Registrar):                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  User Service        Registrar             Service Registry      │   │
│  │       │                 │                       │                │   │
│  │  Start│                 │                       │                │   │
│  │       │◄── Detect ──────│                       │                │   │
│  │       │                 │── Register ──────────►│                │   │
│  │       │                 │                       │                │   │
│  │  Stop │◄── Detect ──────│                       │                │   │
│  │       │                 │── Deregister ────────►│                │   │
│  │                                                                  │   │
│  │  Examples: Netflix Prana, Kubernetes, Docker Swarm              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Major Tools Comparison

| Tool | Type | Features |
|------|------|----------|
| Consul | CP | Health check, KV store, DNS interface |
| Eureka | AP | Netflix OSS, Spring Cloud integration |
| etcd | CP | Raft consensus, Kubernetes-based |
| ZooKeeper | CP | Distributed coordination, complex API |
| Kubernetes | - | Built-in Service + DNS, cloud native |

---

## 2. Circuit Breaker

### Circuit Breaker Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Circuit Breaker Pattern                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem: Cascading Failure                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Order Service → Payment Service → Bank API                     │   │
│  │       │               │              ✗ (failure)                │   │
│  │       │               │                                         │   │
│  │       │         Waiting for timeout...                          │   │
│  │       │         Thread exhaustion                               │   │
│  │       │               ✗                                         │   │
│  │       │                                                         │   │
│  │  Waiting for timeout...                                         │   │
│  │  Thread exhaustion                                              │   │
│  │       ✗                                                         │   │
│  │                                                                  │   │
│  │  → One failure propagates to the entire system!                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Solution: Circuit Breaker (acts like an electrical circuit breaker)   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Order Service → [CB] → Payment Service → [CB] → Bank API       │   │
│  │                          ┌─────────┐        ✗ (failure)         │   │
│  │                          │ Circuit │                             │   │
│  │                          │ OPEN    │ ──► Immediate failure       │   │
│  │                          └─────────┘                             │   │
│  │                                                                  │   │
│  │  → Fast failure protects resources                              │   │
│  │  → Fault isolation                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### State Transitions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Circuit Breaker State Transitions                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                        Failure rate threshold exceeded                  │
│                    ┌────────────────────────┐                          │
│                    │                        │                          │
│                    ▼                        │                          │
│  ┌──────────────────────┐            ┌──────┴───────────────┐          │
│  │                      │            │                      │          │
│  │       CLOSED         │───────────►│        OPEN          │          │
│  │                      │            │                      │          │
│  │  - Normal operation  │            │  - Requests blocked  │          │
│  │  - All requests pass │            │  - Immediate failure │          │
│  │  - Failure rate      │            │  - Fallback executed │          │
│  │    monitoring        │            │                      │          │
│  └──────────────────────┘            └──────────┬───────────┘          │
│           ▲                                     │                      │
│           │                                     │ After timeout        │
│           │                                     ▼                      │
│           │                          ┌──────────────────────┐          │
│           │                          │                      │          │
│           └──────────────────────────│     HALF-OPEN        │          │
│                    Success           │                      │          │
│                                      │  - Limited requests  │          │
│                    Failure           │    allowed           │          │
│                    ┌─────────────────│  - Testing state     │          │
│                    │                 └──────────────────────┘          │
│                    │                            │                      │
│                    └────────────────────────────┘                      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────      │
│                                                                         │
│  Timeline Example:                                                      │
│                                                                         │
│  CLOSED ────────────────────────────► OPEN ──────► HALF-OPEN           │
│  │                                    │              │                  │
│  │ ✓ ✓ ✓ ✗ ✗ ✗ ✗ ✗                   │              │ ✓               │
│  │ Failure rate > 50%                │              │ Success!         │
│  │                                   │ Wait 10s     │                  │
│  │                                   │              ▼                  │
│  │                                   │          CLOSED ───────►        │
│  │                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Configuration Parameters

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Circuit Breaker Configuration                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CircuitBreaker:                                                       │
│    failureRateThreshold: 50        # Failure rate threshold (%)        │
│    slowCallRateThreshold: 100      # Slow call ratio threshold (%)     │
│    slowCallDurationThreshold: 2s   # Slow call duration threshold      │
│    minimumNumberOfCalls: 10        # Minimum calls (for statistics)    │
│    slidingWindowSize: 100          # Sliding window size               │
│    slidingWindowType: COUNT_BASED  # COUNT or TIME_BASED               │
│    waitDurationInOpenState: 10s    # OPEN state duration               │
│    permittedNumberOfCallsInHalfOpen: 3  # Calls allowed in HALF-OPEN   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────      │
│                                                                         │
│  Sliding Window:                                                        │
│                                                                         │
│  COUNT_BASED (based on last N requests):                               │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐                           │
│  │ ✓ │ ✓ │ ✗ │ ✓ │ ✗ │ ✗ │ ✓ │ ✗ │ ✗ │ ✗ │ → Failure rate 60%       │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                           │
│  ◄────────────── Last 10 calls ─────────────►                          │
│                                                                         │
│  TIME_BASED (based on last N seconds):                                 │
│  ┌───────────────────────────────────────┐                             │
│  │  ✓ ✓ ✗   ✓ ✗ ✗   ✓ ✗ ✗ ✗             │ → Failure rate 60%       │
│  └───────────────────────────────────────┘                             │
│  ◄────────────── Last 10 seconds ────────────►                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Fallback Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Fallback Strategies                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Return Default Value                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  def get_user_profile(user_id):                                 │   │
│  │      try:                                                       │   │
│  │          return user_service.get(user_id)                       │   │
│  │      except CircuitOpenException:                               │   │
│  │          return {"name": "Guest", "avatar": "default.png"}     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. Return Cached Data                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  def get_product_price(product_id):                             │   │
│  │      try:                                                       │   │
│  │          price = price_service.get(product_id)                  │   │
│  │          cache.set(product_id, price)                           │   │
│  │          return price                                           │   │
│  │      except CircuitOpenException:                               │   │
│  │          return cache.get(product_id)  # Last cached value     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  3. Call Alternative Service                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  def send_notification(user_id, message):                       │   │
│  │      try:                                                       │   │
│  │          return push_service.send(user_id, message)             │   │
│  │      except CircuitOpenException:                               │   │
│  │          return email_service.send(user_id, message)  # Fallback│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  4. Queue for Later Processing                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  def process_order(order):                                      │   │
│  │      try:                                                       │   │
│  │          return order_service.process(order)                    │   │
│  │      except CircuitOpenException:                               │   │
│  │          retry_queue.enqueue(order)  # Retry later             │   │
│  │          return {"status": "pending"}                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation Tools

| Tool | Language | Features |
|------|----------|----------|
| Resilience4j | Java | Lightweight, functional, modular |
| Hystrix | Java | Netflix OSS (maintenance mode) |
| Polly | .NET | Rich features, policy composition |
| go-kit | Go | Middleware-based |
| opossum | Node.js | Promise support |

---

## 3. Bulkhead Pattern

### Bulkhead Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Bulkhead Pattern                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Ship Bulkheads:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ┌─────┬─────┬─────┬─────┬─────┐                               │   │
│  │  │     │     │Flood│     │     │   ← Separated by bulkheads    │   │
│  │  │  A  │  B  │  C  │  D  │  E  │                               │   │
│  │  │     │     │~~~~~│     │     │   → Even if C compartment     │   │
│  │  └─────┴─────┴─────┴─────┴─────┘     floods, others are safe!  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Software Application:                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Without Bulkheads (Shared Resources):                          │   │
│  │  ┌────────────────────────────────────────────────────────┐     │   │
│  │  │              Thread Pool (10 threads)                  │     │   │
│  │  │  [Payment][Payment][Payment][Payment][Payment]...     │     │   │
│  │  │  All threads occupied by slow Payment calls            │     │   │
│  │  │  → Order, User service calls also blocked!             │     │   │
│  │  └────────────────────────────────────────────────────────┘     │   │
│  │                                                                  │   │
│  │  With Bulkheads (Isolated Resources):                           │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │ Payment Pool │ │  Order Pool  │ │  User Pool   │             │   │
│  │  │  (5 threads) │ │ (3 threads)  │ │ (2 threads)  │             │   │
│  │  │  [P][P][P]   │ │    [O][O]    │ │     [U]      │             │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │   │
│  │  Even if Payment is slow, Order and User work normally!         │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Bulkhead Types

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Bulkhead Implementation Types                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Thread Pool Bulkhead                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Service A Thread Pool                                     │ │   │
│  │  │  maxThreads: 10                                            │ │   │
│  │  │  queueCapacity: 100                                        │ │   │
│  │  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐               │ │   │
│  │  │  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │               │ │   │
│  │  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘               │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  Pros: Complete isolation, easy timeout control                 │   │
│  │  Cons: Overhead (context switching)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. Semaphore Bulkhead                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Service A Semaphore                                       │ │   │
│  │  │  permits: 10  (currently used: 7, waiting: 0)              │ │   │
│  │  │                                                            │ │   │
│  │  │  Request → acquire() → Call → release()                   │ │   │
│  │  │                                                            │ │   │
│  │  │  When permits exceeded → immediate reject or wait          │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  Pros: Lightweight, executes on caller thread                   │   │
│  │  Cons: Difficult timeout control                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────      │
│                                                                         │
│  3. Process-Level Isolation (Containers)                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │  Container  │  │  Container  │  │  Container  │              │   │
│  │  │  Payment    │  │   Order     │  │    User     │              │   │
│  │  │  API calls  │  │  API calls  │  │  API calls  │              │   │
│  │  │  CPU: 2     │  │  CPU: 1     │  │  CPU: 1     │              │   │
│  │  │  Mem: 4GB   │  │  Mem: 2GB   │  │  Mem: 2GB   │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  │                                                                  │   │
│  │  Strongest isolation, Kubernetes Pod resource limits            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Service Mesh

### Service Mesh Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Service Mesh                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem: Each service needs to implement cross-cutting concerns        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Implementing directly in each service:                         │   │
│  │  - Service discovery                                            │   │
│  │  - Load balancing                                               │   │
│  │  - Circuit breaker                                              │   │
│  │  - Retry/timeout                                                │   │
│  │  - TLS/authentication                                           │   │
│  │  - Metrics/tracing                                              │   │
│  │                                                                  │   │
│  │  → Different implementations per language/framework             │   │
│  │  → Difficult to maintain consistency                            │   │
│  │  → Increased developer burden                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Solution: Separate into infrastructure layer (Service Mesh)           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Network proxies handle all cross-cutting concerns              │   │
│  │                                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │                     Control Plane                          │  │   │
│  │  │  (configuration, policies, certificate management)        │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                           ▲ ▲ ▲                                 │   │
│  │                           │ │ │                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │   Service   │  │   Service   │  │   Service   │             │   │
│  │  │   ┌─────┐   │  │   ┌─────┐   │  │   ┌─────┐   │             │   │
│  │  │   │Proxy│◄──┼──┼──►│Proxy│◄──┼──┼──►│Proxy│   │             │   │
│  │  │   └─────┘   │  │   └─────┘   │  │   └─────┘   │             │   │
│  │  │     ▲       │  │     ▲       │  │     ▲       │             │   │
│  │  │     │       │  │     │       │  │     │       │             │   │
│  │  │   ┌─┴─┐     │  │   ┌─┴─┐     │  │   ┌─┴─┐     │             │   │
│  │  │   │App│     │  │   │App│     │  │   │App│     │             │   │
│  │  │   └───┘     │  │   └───┘     │  │   └───┘     │             │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │   │
│  │                                                                  │   │
│  │  App focuses only on business logic!                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Sidecar Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Sidecar Pattern                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Pod (Kubernetes)                                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │                                                             │ │   │
│  │  │  ┌─────────────────────┐   ┌─────────────────────┐         │ │   │
│  │  │  │    Application      │   │      Sidecar        │         │ │   │
│  │  │  │    Container        │   │      Proxy          │         │ │   │
│  │  │  │                     │   │    (Envoy)          │         │ │   │
│  │  │  │  ┌───────────────┐  │   │  ┌───────────────┐  │         │ │   │
│  │  │  │  │  App Process  │◄─┼───┼─►│ Proxy Process │◄─┼─── External │   │
│  │  │  │  │               │  │   │  │               │  │  traffic│ │   │
│  │  │  │  │  Port 8080    │  │   │  │  Port 15001   │  │         │ │   │
│  │  │  │  └───────────────┘  │   │  └───────────────┘  │         │ │   │
│  │  │  │                     │   │                     │         │ │   │
│  │  │  └─────────────────────┘   └─────────────────────┘         │ │   │
│  │  │                                                             │ │   │
│  │  │  Shared: Network Namespace, Volume                          │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Functions handled by sidecar:                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Inbound Traffic:                 Outbound Traffic:             │   │
│  │  - TLS termination               - Service discovery            │   │
│  │  - Authentication/authorization  - Load balancing               │   │
│  │  - Rate limiting                 - Circuit breaker              │   │
│  │  - Metrics collection            - Retry/timeout                │   │
│  │                                   - mTLS                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Istio Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Istio Architecture                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                       Control Plane (istiod)                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │   Pilot     │  │   Citadel   │  │   Galley    │                │ │
│  │  │ (Config     │  │(Certificates│  │  (Config    │                │ │
│  │  │  delivery)  │  │            )│  │ validation) │                │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │ │
│  │         └────────────────┼────────────────┘                        │ │
│  │                          │                                         │ │
│  │                          │ xDS API (config push)                  │ │
│  │                          ▼                                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        Data Plane                                  │ │
│  │                                                                    │ │
│  │  ┌─────────────────┐           ┌─────────────────┐                │ │
│  │  │  Service A      │           │  Service B      │                │ │
│  │  │  ┌───────────┐  │           │  ┌───────────┐  │                │ │
│  │  │  │  Envoy    │◄─┼───────────┼─►│  Envoy    │  │                │ │
│  │  │  │  Proxy    │  │   mTLS    │  │  Proxy    │  │                │ │
│  │  │  └─────┬─────┘  │           │  └─────┬─────┘  │                │ │
│  │  │        │        │           │        │        │                │ │
│  │  │  ┌─────┴─────┐  │           │  ┌─────┴─────┐  │                │ │
│  │  │  │    App    │  │           │  │    App    │  │                │ │
│  │  │  └───────────┘  │           │  └───────────┘  │                │ │
│  │  └─────────────────┘           └─────────────────┘                │ │
│  │                                                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Key Features:                                                         │
│  - Traffic Management: routing, canary deployment, A/B testing         │
│  - Security: mTLS, RBAC                                                │
│  - Observability: metrics, logs, distributed tracing                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Mesh Tools Comparison

| Tool | Proxy | Features |
|------|-------|----------|
| Istio | Envoy | Feature-rich, high complexity |
| Linkerd | linkerd2-proxy | Lightweight, Rust-based, simple |
| Consul Connect | Envoy/built-in | HashiCorp ecosystem integration |
| AWS App Mesh | Envoy | AWS services integration |

---

## 5. Distributed Tracing

### The Need for Distributed Tracing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Distributed Tracing                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem: Tracking requests in microservices                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Client → API GW → Order → Inventory → Payment → Shipping       │   │
│  │                           ↓                                      │   │
│  │                      User Service                                │   │
│  │                                                                  │   │
│  │  "The order API is slow, where is the delay?"                   │   │
│  │  "An error occurred, which service started it?"                 │   │
│  │                                                                  │   │
│  │  Looking at logs:                                                │   │
│  │  - Each service's logs are distributed                          │   │
│  │  - Can't tell which logs belong to the same request             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Solution: Track entire request with Trace ID                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Trace ID: abc123                                               │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ API GW    [──────────────────────────────────────────]    │   │   │
│  │  │ Order     │    [─────────────────────────────────]        │   │   │
│  │  │ Inventory │         [────────────────]                    │   │   │
│  │  │ Payment   │                          [──────────]         │   │   │
│  │  │ User      │    [────]                                     │   │   │
│  │  │           0ms  100ms  200ms  300ms  400ms  500ms         │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                                                                  │   │
│  │  → See entire flow and bottlenecks at a glance!                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Trace, Span, Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Tracing Components                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Trace: The entire journey of a request                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Trace ID: abc-123-def-456                                      │   │
│  │                                                                  │   │
│  │  Span: A single unit of work                                    │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Span A (Root Span)                                        │ │   │
│  │  │  service: api-gateway                                      │ │   │
│  │  │  operation: handle_request                                 │ │   │
│  │  │  ┌──────────────────────────────────────────────────────┐  │ │   │
│  │  │  │  Span B (Child of A)                                  │  │ │   │
│  │  │  │  service: order-service                               │  │ │   │
│  │  │  │  operation: create_order                              │  │ │   │
│  │  │  │  ┌────────────────────────┐ ┌────────────────────┐   │  │ │   │
│  │  │  │  │ Span C (Child of B)    │ │ Span D (Child of B)│   │  │ │   │
│  │  │  │  │ service: inventory     │ │ service: user      │   │  │ │   │
│  │  │  │  │ operation: reserve     │ │ operation: get     │   │  │ │   │
│  │  │  │  └────────────────────────┘ └────────────────────┘   │  │ │   │
│  │  │  └──────────────────────────────────────────────────────┘  │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Span Structure:                                                       │
│  {                                                                      │
│    "traceId": "abc-123-def-456",                                       │
│    "spanId": "span-789",                                               │
│    "parentSpanId": "span-456",                                         │
│    "operationName": "create_order",                                    │
│    "serviceName": "order-service",                                     │
│    "startTime": "2024-01-15T10:30:00.000Z",                           │
│    "duration": 150,  // ms                                             │
│    "tags": { "http.status": 200, "user.id": "123" },                  │
│    "logs": [ { "event": "order_created", "orderId": "ord-999" } ]     │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Context Propagation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Context Propagation                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Propagation via HTTP Headers:                                         │
│                                                                         │
│  Order Service                           Inventory Service              │
│       │                                        │                        │
│       │  POST /inventory/reserve               │                        │
│       │  Headers:                              │                        │
│       │    X-B3-TraceId: abc123               │                        │
│       │    X-B3-SpanId: span456               │                        │
│       │    X-B3-ParentSpanId: span123         │                        │
│       │    X-B3-Sampled: 1                    │                        │
│       │───────────────────────────────────────►│                        │
│       │                                        │                        │
│       │                                        │ Create new Span:       │
│       │                                        │ spanId: span789       │
│       │                                        │ parentSpanId: span456 │
│       │                                        │ traceId: abc123       │
│       │                                        │                        │
│                                                                         │
│  Standards:                                                            │
│  - B3 Propagation (Zipkin)                                             │
│  - W3C Trace Context (standard)                                        │
│  - Jaeger Propagation                                                  │
│                                                                         │
│  W3C Trace Context:                                                    │
│  traceparent: 00-abc123def456-span789-01                               │
│  tracestate: vendor=custom_value                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tracing Tools

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Distributed Tracing Tools                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Jaeger:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - CNCF project                                                 │   │
│  │  - Developed by Uber                                            │   │
│  │  - Cassandra, Elasticsearch backends                            │   │
│  │  - Powerful UI                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Zipkin:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Developed by Twitter                                         │   │
│  │  - Various storage support                                      │   │
│  │  - Lightweight, easy installation                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  OpenTelemetry (OTel):                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Standardized observability framework                         │   │
│  │  - Traces + Metrics + Logs unified                              │   │
│  │  - Vendor neutral                                               │   │
│  │  - Export to Jaeger, Zipkin, etc.                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Commercial:                                                           │
│  - Datadog APM                                                         │
│  - New Relic                                                           │
│  - AWS X-Ray                                                           │
│  - Google Cloud Trace                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Other Important Patterns

### API Gateway

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     API Gateway Pattern                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │   Clients                    API Gateway                        │   │
│  │  ┌──────┐                   ┌─────────────────┐                 │   │
│  │  │ Web  │──────────────────►│                 │                 │   │
│  │  └──────┘                   │  - Routing      │    ┌─────────┐ │   │
│  │  ┌──────┐                   │  - Auth/authz   │───►│ User Svc│ │   │
│  │  │Mobile│──────────────────►│  - Rate limit   │    └─────────┘ │   │
│  │  └──────┘                   │  - Caching      │    ┌─────────┐ │   │
│  │  ┌──────┐                   │  - Request      │───►│Order Svc│ │   │
│  │  │ IoT  │──────────────────►│    transform    │    └─────────┘ │   │
│  │  └──────┘                   │  - Logging/     │    ┌─────────┐ │   │
│  │                             │    metrics      │───►│Prod Svc │ │   │
│  │                             │  - SSL termin.  │    └─────────┘ │   │
│  │                             └─────────────────┘                 │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Tools: Kong, AWS API Gateway, Nginx, Envoy, Spring Cloud Gateway      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Retry Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Retry Strategy                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Exponential Backoff with Jitter:                                      │
│                                                                         │
│  Attempt 1:  ──X (failure)                                             │
│              Wait: 100ms + random(0-50ms)                              │
│  Attempt 2:  ────X (failure)                                           │
│              Wait: 200ms + random(0-100ms)                             │
│  Attempt 3:  ──────X (failure)                                         │
│              Wait: 400ms + random(0-200ms)                             │
│  Attempt 4:  ────────✓ (success)                                       │
│                                                                         │
│  config:                                                                │
│    maxRetries: 5                                                       │
│    initialDelay: 100ms                                                 │
│    maxDelay: 10s                                                       │
│    multiplier: 2                                                       │
│    jitter: 0.5  # 50% random                                           │
│    retryableExceptions:                                                │
│      - ConnectionException                                             │
│      - TimeoutException                                                │
│      # Don't retry 4xx errors!                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Health Check Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Health Check                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Liveness Probe (Is it alive?):                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GET /health/live                                               │   │
│  │                                                                  │   │
│  │  200 OK → Process is healthy                                    │   │
│  │  5xx   → Process needs restart                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Readiness Probe (Can it handle requests?):                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GET /health/ready                                              │   │
│  │                                                                  │   │
│  │  Check items:                                                   │   │
│  │  - DB connection                                                │   │
│  │  - Cache connection                                             │   │
│  │  - Dependent services                                           │   │
│  │  - Initialization complete                                      │   │
│  │                                                                  │   │
│  │  200 OK → Route traffic                                         │   │
│  │  503    → Exclude from traffic (no restart)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Exercise 1: Circuit Breaker Design

Design a circuit breaker for a payment service:
- Set appropriate thresholds
- Define fallback strategies
- Write state transition scenarios

### Exercise 2: Service Mesh Selection

Choose an appropriate service mesh for the following requirements and explain your reasoning:
- 10 microservices
- Kubernetes environment
- mTLS required
- Canary deployment needed
- Team has intermediate Kubernetes experience

### Exercise 3: Distributed Tracing Implementation

Design distributed tracing for an order processing system:
- Define key spans to trace
- Define important tags/metadata
- Establish sampling strategy

---

**Previous**: [Microservices Fundamentals](./13_Microservices_Basics.md) | **Next**: [Distributed Systems Concepts](./15_Distributed_Systems_Concepts.md)

---

## References

- "Release It!" - Michael Nygard
- "Building Microservices" - Sam Newman
- Istio Documentation
- Envoy Proxy Documentation
- OpenTelemetry Documentation
- Netflix Tech Blog: Hystrix
- Resilience4j Documentation
