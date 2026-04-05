# Chaos Engineering

**Previous**: [Secrets Management](./15_Secrets_Management.md) | **Next**: [Platform Engineering](./17_Platform_Engineering.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define chaos engineering principles and explain why proactive failure injection builds more resilient systems
2. Design experiments using the scientific method: steady-state hypothesis, experiment, observation, conclusion
3. Implement failure injection using Chaos Monkey, Litmus, and other chaos engineering tools
4. Control blast radius to prevent experiments from causing real outages
5. Plan and execute game days to practice incident response under controlled conditions
6. Classify failure injection types and select appropriate experiments for different system components

---

Chaos engineering is the discipline of experimenting on a system to build confidence in its ability to withstand turbulent conditions in production. Instead of waiting for failures to happen at 3 AM, chaos engineers deliberately inject failures during business hours when teams are ready to respond. The goal is not to break things -- it is to find weaknesses before they find you.

> **Analogy -- Fire Drills**: A fire drill does not set the building on fire. It simulates an emergency to test whether people know the evacuation routes, whether alarms work, and whether exits are unblocked. Chaos engineering is a fire drill for your distributed system: you simulate failures (server crash, network partition, disk full) to verify that your monitoring detects the problem, your automation recovers, and your users are not affected.

## 1. Why Chaos Engineering

### 1.1 The Resilience Gap

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

Common assumptions that chaos engineering disproves:
- "Our auto-scaling handles traffic spikes" (it does not if the scaling policy has a bug)
- "Our circuit breaker prevents cascading failures" (it does not if the timeout is set too high)
- "Our database failover is seamless" (it is not if the application does not reconnect)
- "Our monitoring catches all issues" (it does not if alerts are misconfigured)

### 1.2 Chaos Engineering vs Testing

| Aspect | Traditional Testing | Chaos Engineering |
|--------|-------------------|-------------------|
| **Environment** | Test/staging | Production (preferred) |
| **Scope** | Known failure modes | Unknown failure modes |
| **Goal** | Verify expected behavior | Discover unexpected behavior |
| **Methodology** | Test cases with pass/fail | Experiments with hypotheses |
| **Failures** | Simulated/mocked | Real (injected into live systems) |

---

## 2. Principles of Chaos Engineering

### 2.1 The Five Principles

1. **Build a hypothesis around steady-state behavior**
   - Define what "normal" looks like using metrics (error rate < 0.1%, p99 latency < 500ms)
   - The hypothesis is: "When we inject failure X, the steady state will be maintained"

2. **Vary real-world events**
   - Inject failures that actually happen: server crashes, network partitions, disk full, high CPU
   - Do not invent unrealistic scenarios

3. **Run experiments in production**
   - Staging environments do not have real traffic patterns, data volumes, or dependency behaviors
   - Production experiments reveal real behavior (with safety controls)

4. **Automate experiments to run continuously**
   - One-off experiments prove a point in time; continuous experiments catch regressions
   - Integrate chaos experiments into CI/CD pipelines

5. **Minimize blast radius**
   - Start small (one instance, one AZ, canary traffic)
   - Have a kill switch to stop experiments immediately
   - Gradually expand scope as confidence grows

### 2.2 The Experiment Lifecycle

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

## 3. Failure Injection Types

### 3.1 Classification

| Category | Failure Type | Example | Tools |
|----------|-------------|---------|-------|
| **Infrastructure** | Instance termination | Kill a VM or pod | Chaos Monkey, Litmus |
| **Infrastructure** | AZ failure | Simulate entire zone outage | AWS FIS, Gremlin |
| **Network** | Latency injection | Add 500ms delay to service calls | tc, Toxiproxy, Istio |
| **Network** | Packet loss | Drop 10% of network packets | tc, Litmus |
| **Network** | DNS failure | DNS resolution fails | CoreDNS manipulation |
| **Network** | Partition | Service A cannot reach Service B | iptables, Istio fault injection |
| **Resource** | CPU stress | Consume 95% CPU | stress-ng, Litmus |
| **Resource** | Memory pressure | Consume available memory until OOM | stress-ng, Litmus |
| **Resource** | Disk fill | Fill disk to 100% | dd, Litmus |
| **Application** | Exception injection | Force specific code paths to throw errors | Feature flags, SDK |
| **Application** | Dependency failure | External API returns 500 or times out | Toxiproxy, Istio |
| **State** | Clock skew | Set system clock forward/backward | chrony manipulation |
| **State** | Data corruption | Invalid data in message queue or cache | Custom scripts |

### 3.2 Network Failure Injection

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

### 4.1 Netflix's Chaos Monkey

Chaos Monkey is Netflix's tool that randomly terminates instances in production. It was the first widely known chaos engineering tool and established the principle that systems should be designed to tolerate individual instance failures.

**Key concepts:**
- Runs during business hours (when engineers are available to respond)
- Targets a random instance in a configured group
- Forces teams to design for instance failure from day one

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

### 5.1 Architecture

Litmus is a CNCF project that provides a Kubernetes-native chaos engineering framework:

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

### 5.2 Pod Delete Experiment

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

### 5.3 Network Chaos Experiment

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

## 6. Steady-State Hypothesis

### 6.1 Defining Steady State

The steady-state hypothesis is the most critical part of a chaos experiment. It defines "normal" in measurable terms:

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

### 6.2 Abort Criteria

Define when to automatically stop an experiment:

| Metric | Normal | Warning (Continue) | Abort (Stop Immediately) |
|--------|--------|--------------------|-----------------------------|
| Error rate | < 0.1% | 0.1% - 1% | > 1% |
| p99 latency | < 500ms | 500ms - 2s | > 2s |
| Pod restarts | 0 | 1-2 | > 2 |
| Customer impact | None | None visible | Any user-facing error |

---

## 7. Blast Radius Control

### 7.1 Progressive Blast Radius

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

### 7.2 Safety Controls

| Control | Description |
|---------|-------------|
| **Kill switch** | Manual button to instantly stop all chaos experiments |
| **Time limits** | Experiments auto-stop after a maximum duration |
| **Metric guards** | Abort if key metrics breach thresholds |
| **Namespace isolation** | Chaos experiments only target allowed namespaces |
| **Business hours** | Run only when engineering team is available (9 AM - 4 PM) |
| **Change freeze** | Do not run during deployments, maintenance, or incidents |
| **Opt-in labeling** | Only target resources with explicit chaos labels |

---

## 8. Game Days

### 8.1 What is a Game Day

A game day is a scheduled, team-wide chaos engineering event where the team deliberately injects failures into production (or production-like environments) and practices incident response.

### 8.2 Game Day Structure

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

### 8.3 Example Game Day Scenarios

| Scenario | Hypothesis | What To Observe |
|----------|-----------|-----------------|
| **Kill 50% of API pods** | Kubernetes reschedules pods within 30s; error rate stays below 0.5% | Pod scheduling time, request errors during transition |
| **Database primary failover** | RDS multi-AZ failover completes in < 60s; app reconnects automatically | Failover duration, connection pool recovery, data consistency |
| **Redis cache cluster down** | Application falls back to database queries; latency increases but no errors | Cache miss rate, DB query load, response times |
| **DNS failure** | Circuit breaker opens; cached responses are served; alert fires within 2 minutes | Circuit breaker behavior, monitoring detection time |
| **Region failure** | Traffic fails over to secondary region within 5 minutes; data is consistent | DNS failover time, data replication lag, user impact |

---

## 9. Chaos Engineering Maturity Model

### 9.1 Maturity Levels

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

## 10. Next Steps

- [17_Platform_Engineering.md](./17_Platform_Engineering.md) - Building platforms that are chaos-resilient by design
- [18_SRE_Practices.md](./18_SRE_Practices.md) - SRE practices for managing reliability budgets

---

## Exercises

### Exercise 1: Experiment Design

Design a chaos experiment for the following scenario:

Your e-commerce platform has a checkout service that depends on an inventory service, a payment service, and a notification service. The inventory and payment services are critical (checkout fails without them), but the notification service is non-critical (email confirmation can be delayed).

Design an experiment to verify that the checkout service handles a notification service outage gracefully.

<details>
<summary>Show Answer</summary>

**Experiment: Notification Service Outage During Checkout**

**Steady-state hypothesis:**
- Checkout success rate > 99.5%
- Checkout p99 latency < 2 seconds
- Orders per minute > baseline (current rate)
- No user-facing errors on the checkout page

**Hypothesis statement:**
"When the notification service is completely unavailable, checkout operations will complete successfully. Email notifications will be queued and delivered when the service recovers. Users will see a success page (not an error page)."

**Experiment design:**
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

**Blast radius controls:**
- Only affects the notification service (checkout, inventory, payment are untouched)
- Duration: 5 minutes maximum
- Abort if checkout success rate drops below 99% (not 99.5% -- allows a small margin)
- Run during business hours with the on-call engineer aware

**Expected outcomes:**
1. **If the system is well-designed**: Checkout calls to the notification service use an async pattern (message queue). The message is queued even if the consumer is down. Checkout succeeds immediately. When the notification service recovers, it processes the queued messages and sends emails.
2. **If there is a bug**: The checkout service makes a synchronous HTTP call to the notification service. The call times out after 5 seconds, and the checkout fails with a 500 error. **This is the bug to fix**: the notification call should be asynchronous with a circuit breaker.

**Fix (if the experiment fails):**
- Change the notification call from synchronous HTTP to an async message queue (SQS, RabbitMQ)
- Add a circuit breaker with a fallback that logs the notification for later delivery
- Re-run the experiment to verify the fix

</details>

### Exercise 2: Blast Radius Assessment

For each of the following chaos experiments, assess the blast radius (low, medium, high) and identify what could go wrong if the experiment lacks proper safety controls:

1. Killing 1 of 10 pods in a stateless API service
2. Injecting 2-second network latency between the API gateway and all backend services
3. Filling the disk to 100% on the primary database server
4. Simulating a complete AWS region failure

<details>
<summary>Show Answer</summary>

**1. Kill 1 of 10 pods — Blast radius: LOW**
- Impact: 10% capacity reduction; load balancer routes around the dead pod
- Risk without controls: Minimal. Kubernetes reschedules within seconds. If the pod has a long startup time, brief capacity reduction.
- What could go wrong: If the pod holds in-memory state (sticky sessions, local cache), those sessions are lost. If the deployment has a pod disruption budget (PDB) set incorrectly, killing might be blocked.
- Safety: Standard readiness probes and PDBs are sufficient.

**2. 2-second latency to all backends — Blast radius: HIGH**
- Impact: Every request through the API gateway takes an additional 2 seconds. All users experience degraded performance.
- Risk without controls: Connection pool exhaustion. The API gateway holds connections open for 2+ seconds instead of the normal 50ms, rapidly exhausting the connection pool. Thread pool exhaustion follows, and the gateway starts rejecting all requests. Cascading failure: upstream services (CDN, load balancer) start timing out.
- What could go wrong: If the experiment does not have an automatic abort on latency threshold, the entire platform could become unavailable.
- Safety: Never inject latency to ALL backends simultaneously. Start with one backend at a time, and use a percentage (10%) not 100%.

**3. Fill disk to 100% on primary database — Blast radius: HIGH (DANGEROUS)**
- Impact: Database cannot write. All write operations fail. Transactions are rolled back. WAL (write-ahead log) cannot be written, which may cause data corruption.
- Risk without controls: **Data loss**. If the database cannot write WAL, it may need to be restored from backup. Replication may break. Recovery could take hours.
- What could go wrong: This is one of the most dangerous experiments. Unlike killing a pod (which is stateless), disk fill on a database can cause permanent data loss.
- Safety: **Never run this on a production database.** Test on a staging database with a replica. If you must test in production, fill a non-critical volume (e.g., /tmp), not the data directory.

**4. Simulate AWS region failure — Blast radius: VERY HIGH**
- Impact: All services in that region are unavailable. Only services with multi-region redundancy survive.
- Risk without controls: Complete outage for all users if multi-region failover does not work correctly. DNS propagation delays could extend the outage to 10+ minutes even if the failover is correct.
- What could go wrong: Data inconsistency if the database replication lag is non-zero. DNS caches may continue routing to the failed region. CDN edge caches may become stale.
- Safety: This is a game day-level experiment requiring full team readiness, pre-validated rollback procedures, and stakeholder notification. Never run ad-hoc.

</details>

### Exercise 3: Steady-State Hypothesis

Write a complete steady-state hypothesis for a payment processing service that:
- Processes credit card charges via Stripe
- Must maintain 99.99% success rate
- Has a p99 latency SLO of 1 second
- Stores transactions in PostgreSQL

Include the metrics, thresholds, and monitoring queries.

<details>
<summary>Show Answer</summary>

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
      threshold: ">= 50"     # Minimum expected throughput
      alert_on_breach: true
      severity: warning

    # Latency metrics
    - name: payment_p99_latency
      description: "99th percentile payment processing latency"
      query: >
        histogram_quantile(0.99,
          sum by (le) (rate(payment_duration_seconds_bucket[5m]))
        )
      threshold: "< 1.0"      # 1 second SLO
      alert_on_breach: true
      severity: critical

    - name: payment_p50_latency
      description: "50th percentile payment processing latency"
      query: >
        histogram_quantile(0.50,
          sum by (le) (rate(payment_duration_seconds_bucket[5m]))
        )
      threshold: "< 0.3"      # 300ms expected median
      alert_on_breach: false
      severity: info

    # Infrastructure metrics
    - name: db_connection_pool_utilization
      description: "PostgreSQL connection pool usage"
      query: >
        payment_db_pool_active_connections
        / payment_db_pool_max_connections
      threshold: "< 0.80"     # Below 80% utilization
      alert_on_breach: true
      severity: warning

    - name: db_replication_lag
      description: "PostgreSQL replication lag in seconds"
      query: "pg_replication_lag_seconds{instance='payment-db'}"
      threshold: "< 1.0"      # Less than 1 second
      alert_on_breach: true
      severity: critical

    - name: stripe_api_error_rate
      description: "Stripe API call failure rate"
      query: >
        sum(rate(stripe_api_calls_total{status="error"}[5m]))
        / sum(rate(stripe_api_calls_total[5m]))
      threshold: "< 0.01"     # Less than 1% Stripe errors
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
      threshold: "== 0"       # No restarts
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

### Exercise 4: Game Day Planning

Plan a game day for a team of 8 engineers. The system is a 3-tier web application (frontend, API, database) running on Kubernetes with AWS RDS. Design 4 experiments with increasing blast radius, define roles, and create a communication plan.

<details>
<summary>Show Answer</summary>

**Game Day: "Operation Resilience" -- Q1 2024**

**Date**: Tuesday, 10:00 AM - 2:00 PM (during business hours, mid-week)

**Roles (8 engineers):**

| Role | Person | Responsibility |
|------|--------|----------------|
| **Game Day Lead** | Engineer 1 | Coordinates experiments, makes go/no-go decisions |
| **Experiment Operator** | Engineer 2 | Executes chaos experiments, controls kill switch |
| **Metrics Observer** | Engineer 3 | Monitors dashboards, reports metric changes in real-time |
| **Log Observer** | Engineer 4 | Watches logs for errors, correlates with experiments |
| **Customer Impact Monitor** | Engineer 5 | Monitors synthetic tests and customer-facing error pages |
| **Incident Commander** | Engineer 6 | Takes over if experiment causes real incident |
| **Scribe** | Engineer 7 | Documents everything: timeline, observations, surprises |
| **Stakeholder Liaison** | Engineer 8 | Communicates with support team and leadership |

**Experiments (increasing blast radius):**

| # | Time | Experiment | Blast Radius | Hypothesis |
|---|------|-----------|-------------|-----------|
| 1 | 10:30 | Kill 1 of 4 API pods | Low | K8s reschedules in < 30s; no user errors |
| 2 | 11:00 | Inject 500ms latency between API and DB | Medium | Circuit breaker opens at 1s; cached responses served; p99 < 2s |
| 3 | 11:30 | Kill all Redis cache pods | Medium-High | API falls back to DB; latency increases 3x but no errors |
| 4 | 12:00 | Simulate RDS failover (Multi-AZ) | High | Failover completes < 60s; app reconnects; < 5 failed transactions |

**Communication plan:**

| When | Channel | Message |
|------|---------|---------|
| 1 week before | Email to engineering + support | "Game Day scheduled for [date]. No customer impact expected. Support team: be aware of potential brief latency increases." |
| Morning of | Slack #game-day channel | "Game Day starting at 10:00. All experiments will be announced here. Kill switch holder: [Engineer 2]." |
| Before each experiment | Slack #game-day | "Starting Experiment [N]: [description]. Blast radius: [level]. Abort criteria: [thresholds]." |
| After each experiment | Slack #game-day | "Experiment [N] complete. Result: [pass/fail]. Key observation: [summary]." |
| End of day | Email to engineering + leadership | "Game Day summary: 4 experiments, [N] passed, [N] revealed issues. Action items: [list]." |

**Abort protocol:**
1. Any engineer can call "ABORT" at any time
2. Experiment Operator immediately stops the experiment (kill switch)
3. Incident Commander assesses whether recovery is needed
4. Team decides whether to continue with next experiment or end the game day

**Pre-requisites:**
- [ ] All monitoring dashboards loaded and shared
- [ ] Kill switch tested in staging
- [ ] RDS failover tested in staging
- [ ] Support team briefed
- [ ] Rollback procedures documented for each experiment
- [ ] No other deployments or maintenance scheduled

</details>

---

## References

- [Principles of Chaos Engineering](https://principlesofchaos.org/)
- [Netflix Chaos Monkey](https://netflix.github.io/chaosmonkey/)
- [Litmus Chaos Documentation](https://litmuschaos.io/)
- [AWS Fault Injection Simulator](https://docs.aws.amazon.com/fis/)
- [Gremlin Documentation](https://www.gremlin.com/docs/)
- [Chaos Engineering Book (O'Reilly)](https://www.oreilly.com/library/view/chaos-engineering/9781492043850/)
