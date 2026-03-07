# Deployment Strategies

**Previous**: [Distributed Tracing](./12_Distributed_Tracing.md) | **Next**: [GitOps](./14_GitOps.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare deployment strategies (blue-green, canary, rolling, A/B testing) and select the appropriate one based on risk tolerance and infrastructure constraints
2. Implement blue-green deployments with instant rollback using load balancer switching
3. Design canary release pipelines with automated metric-based promotion and rollback
4. Configure rolling update parameters in Kubernetes to balance deployment speed and availability
5. Distinguish between A/B testing (user experiments) and canary releases (risk mitigation)
6. Implement feature flags using LaunchDarkly or Unleash for decoupling deployment from release

---

Deploying new code to production is the most dangerous routine activity in software operations. Every deployment risks introducing bugs, performance regressions, or outages. Deployment strategies exist to reduce that risk by controlling how new code reaches users -- gradually, safely, and with the ability to roll back instantly. This lesson covers the major strategies, their trade-offs, and practical implementation patterns.

> **Analogy -- Testing a New Bridge**: Imagine a city needs to replace an old bridge. A **rolling update** is repairing one lane at a time while traffic uses the remaining lanes. A **blue-green deployment** is building a second bridge alongside the old one and switching all traffic at once. A **canary release** is routing 5% of traffic across the new bridge first and expanding only if it holds. **A/B testing** is sending trucks across the new bridge and cars across the old one to measure which handles each type better.

## 1. Strategy Overview

### 1.1 Comparison Matrix

| Strategy | Risk | Rollback Speed | Infrastructure Cost | Zero Downtime | Best For |
|----------|------|---------------|--------------------|--------------|---------|
| **Rolling Update** | Medium | Slow (minutes) | Low (in-place) | Yes | Kubernetes default, stateless services |
| **Blue-Green** | Low | Instant (seconds) | High (2x capacity) | Yes | Critical services, database migrations |
| **Canary** | Very Low | Fast (seconds) | Medium (N+few) | Yes | High-traffic services, gradual validation |
| **A/B Testing** | Low | Fast | Medium | Yes | Feature experiments, user research |
| **Recreate** | High | Slow | Low | **No** | Dev/staging, legacy monoliths |

### 1.2 Visual Comparison

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

## 2. Rolling Updates

### 2.1 How Rolling Updates Work

Rolling updates replace instances one (or a few) at a time. At any point during the deployment, both old and new versions coexist.

```
Time 0:  [v1] [v1] [v1] [v1]     ← All old
Time 1:  [v2] [v1] [v1] [v1]     ← 1 new, 3 old
Time 2:  [v2] [v2] [v1] [v1]     ← 2 new, 2 old
Time 3:  [v2] [v2] [v2] [v1]     ← 3 new, 1 old
Time 4:  [v2] [v2] [v2] [v2]     ← All new
```

### 2.2 Kubernetes Rolling Update Configuration

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

**Key parameters explained:**

| Parameter | Effect | Recommended |
|-----------|--------|------------|
| `maxSurge: 1` | At most 1 extra pod during rollout | Keep low for resource-constrained clusters |
| `maxUnavailable: 0` | Never reduce below desired replica count | Use `0` for critical services |
| `minReadySeconds: 30` | Pod must be ready for 30s before rollout continues | Prevents fast rollout of broken versions |
| `readinessProbe` | Pod receives traffic only when ready | Always configure for rolling updates |

### 2.3 Rolling Update Commands

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

### 2.4 Limitations of Rolling Updates

- **Mixed versions**: During rollout, both v1 and v2 serve traffic simultaneously. APIs must be backward-compatible.
- **Slow rollback**: Rollback is another rolling update (replacing v2 with v1, one pod at a time).
- **No traffic control**: You cannot route specific users or a percentage of traffic to v2.

---

## 3. Blue-Green Deployments

### 3.1 How Blue-Green Works

Two identical environments (blue and green) exist. Only one serves production traffic at any time. Deployments go to the inactive environment, and a load balancer switch cuts over all traffic at once.

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

### 3.2 Implementation with Kubernetes Services

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

### 3.3 Database Considerations

The hardest part of blue-green deployments is the database. Both blue and green must be compatible with the same database schema.

**Safe migration pattern:**
1. **Expand**: Add new columns/tables (backward-compatible) -- both v1 and v2 work
2. **Migrate**: Deploy v2 (green) and switch traffic
3. **Contract**: Remove old columns/tables after v1 (blue) is decommissioned

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

## 4. Canary Releases

### 4.1 How Canary Releases Work

A canary release routes a small percentage of production traffic to the new version. Metrics are monitored, and the percentage is gradually increased if the canary is healthy. If metrics degrade, the canary is rolled back.

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

### 4.2 Canary with Istio Service Mesh

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

### 4.3 Automated Canary with Flagger

Flagger automates canary analysis and promotion in Kubernetes:

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

**Flagger progression:**
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

| Aspect | Canary Release | A/B Testing |
|--------|---------------|-------------|
| **Goal** | Reduce deployment risk | Measure feature impact on user behavior |
| **Routing** | Random percentage of traffic | Specific user segments (cohorts) |
| **Metrics** | Error rate, latency, saturation | Conversion rate, revenue, engagement |
| **Duration** | Minutes to hours | Days to weeks |
| **Decision** | Automated (metric thresholds) | Data science analysis (statistical significance) |

### 5.2 A/B Testing with Header-Based Routing

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

### 5.3 A/B Testing Architecture

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

## 6. Feature Flags

### 6.1 What are Feature Flags

Feature flags (feature toggles) decouple deployment from release. Code for a new feature is deployed to production but hidden behind a flag. The flag controls who sees the feature without redeploying.

```
Traditional:  Code Change → Deploy → Release (all at once)

Feature Flag: Code Change → Deploy (flag OFF) → Release to 5% → Release to 50% → Release to 100%
                                                      ↑                  ↑              ↑
                                                  Flag toggle        Flag toggle    Remove flag
```

### 6.2 Feature Flag Types

| Type | Lifespan | Purpose | Example |
|------|----------|---------|---------|
| **Release toggle** | Days to weeks | Gradual rollout of new features | New checkout flow |
| **Experiment toggle** | Weeks to months | A/B testing | Button color experiment |
| **Ops toggle** | Permanent | Kill switch for degraded mode | Disable recommendations under load |
| **Permission toggle** | Permanent | Feature access by user role/plan | Premium feature for paid users |

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

### 6.4 Unleash (Open-Source Alternative)

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

### 6.5 Feature Flag Best Practices

| Practice | Description |
|----------|-------------|
| **Short-lived flags** | Remove release toggles within 2 weeks of full rollout |
| **Flag naming** | Use descriptive names: `enable-new-checkout-flow`, not `flag-123` |
| **Default to off** | New flags should default to `false` (safe by default) |
| **Document flags** | Track owner, creation date, expected removal date |
| **Test both paths** | Unit tests must cover both flag on and flag off |
| **Monitoring** | Alert when a flag evaluation fails (SDK outage) |
| **Clean up** | Schedule flag removal as tech debt; stale flags add complexity |

---

## 7. Dark Launches

### 7.1 What is a Dark Launch

A dark launch deploys new functionality to production and routes real production traffic to it, but the results are discarded (not shown to users). This tests the new code under real load without user impact.

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

### 7.2 Traffic Mirroring with Istio

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

### 7.3 Use Cases for Dark Launches

| Use Case | Description |
|----------|-------------|
| **Performance testing** | Verify v2 handles production load without impacting users |
| **Data validation** | Compare v2 responses to v1 responses (diff analysis) |
| **Database migration** | Shadow-write to new database and compare results |
| **ML model validation** | Run new model on production data, compare predictions |

---

## 8. Strategy Selection Guide

### 8.1 Decision Framework

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

### 8.2 Strategy by Service Type

| Service Type | Recommended Strategy | Reasoning |
|-------------|---------------------|-----------|
| **Stateless API** | Canary or Rolling Update | Easy to scale, no session state |
| **Stateful service** | Blue-Green | Database compatibility needs careful cutover |
| **Frontend/SPA** | Blue-Green or Feature Flags | Users should not see mixed UI versions |
| **Background workers** | Rolling Update | No user-facing traffic, lower risk |
| **Data pipeline** | Blue-Green | Must validate output before switching |
| **Mobile app** | Feature Flags | Cannot force-update user devices |

---

## 9. Next Steps

- [14_GitOps.md](./14_GitOps.md) - Declarative deployment with ArgoCD and Flux
- [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) - Validating deployment resilience

---

## Exercises

### Exercise 1: Strategy Selection

For each scenario, recommend a deployment strategy and justify your choice:

1. An e-commerce company is deploying a new payment gateway integration. A bug could result in failed charges and lost revenue.
2. A social media platform wants to test whether a new algorithmic feed increases user engagement compared to a chronological feed.
3. A startup with limited infrastructure budget needs to deploy updates to a stateless REST API.
4. A bank needs to migrate from Oracle to PostgreSQL for its core banking application.

<details>
<summary>Show Answer</summary>

1. **Canary release** -- Payment processing is high-risk. A canary routes 1-5% of payment traffic to the new gateway while monitoring success rate, latency, and error rate. If the canary shows higher failure rates, it is automatically rolled back before most users are affected. Blue-green would also work, but canary provides more gradual risk reduction.

2. **A/B testing with feature flags** -- This is a user behavior experiment, not a deployment risk question. Assign users to control (chronological) and variant (algorithmic) cohorts using a feature flag. Run the experiment for 2-4 weeks to reach statistical significance. Measure engagement metrics (time on site, posts read, return visits). A canary release would not work here because it routes random traffic, not persistent user cohorts.

3. **Rolling update** -- For a stateless REST API on limited infrastructure, rolling updates are the best fit. They require no extra infrastructure (no second environment like blue-green), are built into Kubernetes by default, and provide zero-downtime deployments. The trade-off is slower rollback, but for a startup with lower traffic, this is acceptable.

4. **Blue-green deployment with dark launch** -- First, run a dark launch: mirror all production traffic to the PostgreSQL environment and compare query results with Oracle. Once parity is verified, use blue-green to cut over. Blue-green allows instant rollback to Oracle if issues are discovered after switching. This is the only strategy that provides instant rollback for a database migration -- canary would result in split-brain issues with two databases.

</details>

### Exercise 2: Kubernetes Rolling Update

A deployment has 4 replicas and is configured with `maxSurge: 1` and `maxUnavailable: 1`. Describe the step-by-step rollout process when updating from v1 to v2, including how many pods exist at each step and how many serve traffic.

<details>
<summary>Show Answer</summary>

With `replicas: 4`, `maxSurge: 1`, `maxUnavailable: 1`:
- Max pods at any time: 4 + 1 = 5
- Min available pods at any time: 4 - 1 = 3

**Step-by-step:**

| Step | v1 Pods | v2 Pods | Total | Available | Action |
|------|---------|---------|-------|-----------|--------|
| 0 | 4 running | 0 | 4 | 4 | Initial state |
| 1 | 3 running, 1 terminating | 1 creating | 5 | 3 | Scale up v2, scale down v1 simultaneously |
| 2 | 3 running | 1 ready | 4 | 4 | v2 pod passes readiness probe |
| 3 | 2 running, 1 terminating | 1 ready, 1 creating | 4 | 3 | Next v1 pod terminates, next v2 pod starts |
| 4 | 2 running | 2 ready | 4 | 4 | Second v2 pod passes readiness probe |
| 5 | 1 running, 1 terminating | 2 ready, 1 creating | 4 | 3 | Continue pattern |
| 6 | 1 running | 3 ready | 4 | 4 | Third v2 pod ready |
| 7 | 0 (terminating) | 3 ready, 1 creating | 4 | 3 | Last v1 pod terminates |
| 8 | 0 | 4 ready | 4 | 4 | Rollout complete |

**Key observations:**
- The system always has at least 3 available pods (satisfying `maxUnavailable: 1`).
- The system never exceeds 5 total pods (satisfying `maxSurge: 1`).
- Both `maxSurge` and `maxUnavailable` being non-zero allows the rollout to proceed faster by simultaneously creating new pods and terminating old ones.

</details>

### Exercise 3: Canary Metrics

You are deploying a canary for an API service. Define the metrics you would use to evaluate the canary, the thresholds for automatic promotion, and the thresholds for automatic rollback.

<details>
<summary>Show Answer</summary>

**Canary evaluation metrics:**

| Metric | Promotion Threshold | Rollback Threshold | Source |
|--------|--------------------|--------------------|--------|
| **Error rate** | < 0.5% (same as stable) | > 1% or > 2x stable | Prometheus: `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])` |
| **p99 latency** | < 500ms | > 1000ms or > 2x stable | Prometheus: `histogram_quantile(0.99, ...)` |
| **p50 latency** | < 100ms | > 200ms or > 2x stable | Prometheus: `histogram_quantile(0.50, ...)` |
| **CPU usage** | < 80% | > 95% | Prometheus: container CPU metrics |
| **Memory usage** | < 80% | > 90% | Prometheus: container memory metrics |
| **Panic/restart count** | 0 | > 0 | Kubernetes: `kube_pod_container_status_restarts_total` |

**Evaluation approach:**
1. Compare canary metrics to the stable version (not absolute thresholds alone).
2. Use relative comparison: "canary error rate < 1.5x stable error rate" catches regressions even if absolute rates are low.
3. Require all metrics to pass for promotion. Any single metric failure triggers rollback.
4. Wait at least 5 minutes at each weight step to collect statistically meaningful data.

**Flagger metric template:**
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

### Exercise 4: Feature Flag Migration

Your team has accumulated 47 feature flags in production, some over a year old. Design a plan to clean up stale flags and prevent future accumulation.

<details>
<summary>Show Answer</summary>

**Immediate cleanup plan:**

1. **Audit all 47 flags**: Categorize each as:
   - **Fully rolled out** (flag is ON for 100% of users for > 2 weeks): Remove the flag and the old code path.
   - **Abandoned** (flag has not been changed in > 3 months): Contact the owner. If no response in 1 week, remove the flag (keep the current default behavior).
   - **Active experiment**: Keep, but set a review date.
   - **Ops toggle**: Keep (permanent by design), document purpose.

2. **Prioritize removal**: Start with flags that are `ON for 100%` since they are risk-free to remove (the new code path is already proven).

3. **Code changes**: For each removed flag, delete:
   - The flag evaluation code
   - The old code path (dead code)
   - Related tests for the old path
   - The flag definition in the flag management platform

**Prevention policy:**

| Rule | Description |
|------|-------------|
| **Expiration date** | Every release toggle must have a removal date (max 30 days after full rollout) |
| **Owner assignment** | Every flag has an assigned owner responsible for cleanup |
| **Automated alerts** | Alert the owner when a flag has been 100% ON for > 14 days |
| **CI check** | Fail the build if total flag count exceeds a threshold (e.g., 20) |
| **Review process** | Monthly flag review meeting (15 minutes) to identify stale flags |
| **Naming convention** | Include creation date in flag name: `2024-03-checkout-v2` |

**Metric to track**: "Flag debt" = number of flags older than 30 days that are 100% on. Target: 0.

</details>

---

## References

- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Flagger - Progressive Delivery](https://flagger.app/)
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [LaunchDarkly Documentation](https://docs.launchdarkly.com/)
- [Unleash Documentation](https://docs.getunleash.io/)
- [Martin Fowler - Feature Toggles](https://martinfowler.com/articles/feature-toggles.html)
