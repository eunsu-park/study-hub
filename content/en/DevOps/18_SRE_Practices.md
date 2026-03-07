# SRE Practices

**Previous**: [Platform Engineering](./17_Platform_Engineering.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define Site Reliability Engineering (SRE) and explain how it relates to and differs from DevOps
2. Design SLOs, SLIs, and SLAs for services and use them to make engineering decisions
3. Calculate and manage error budgets to balance reliability with feature velocity
4. Identify and eliminate toil through automation and engineering solutions
5. Implement structured incident management with clear roles, communication, and escalation
6. Conduct effective postmortems that produce actionable improvements without blame
7. Design sustainable on-call practices that respect engineer well-being

---

Site Reliability Engineering (SRE) is a discipline that applies software engineering principles to infrastructure and operations. Coined at Google in 2003, SRE answers the question: "What happens when you ask a software engineer to design an operations team?" The answer is a set of practices that treat reliability as a feature, define it with precise metrics (SLOs), and use error budgets to make principled trade-offs between reliability and velocity.

> **Analogy -- Bridge Engineering**: A civil engineer does not build a bridge that never fails under any load -- that would be infinitely expensive and take forever. Instead, they define a load specification (SLO): "This bridge must safely support 10,000 vehicles per day with a safety factor of 2x." They measure actual load (SLI) and track the safety margin (error budget). If the bridge is consistently at 80% capacity, they can approve adding a bus route. If it is at 98%, they halt new traffic until reinforcements are added. SRE applies this same engineering discipline to software systems.

## 1. SRE vs DevOps

### 1.1 Relationship

```
DevOps:  A cultural movement and set of practices
         ┌─────────────────────────────────────┐
         │  • Break down silos                 │
         │  • Automate everything              │
         │  • Continuous improvement            │
         │  • Shared responsibility             │
         └─────────────────────────────────────┘
                         │
                         │ "class SRE implements DevOps"
                         ▼
SRE:     A specific implementation with concrete practices
         ┌─────────────────────────────────────┐
         │  • SLOs/SLIs/SLAs (measurable)      │
         │  • Error budgets (decision framework)│
         │  • Toil reduction (< 50% rule)       │
         │  • Incident management (structured)  │
         │  • Postmortems (blameless)           │
         │  • On-call (sustainable)             │
         └─────────────────────────────────────┘
```

### 1.2 Key Differences

| Aspect | DevOps | SRE |
|--------|--------|-----|
| **Origin** | Grassroots cultural movement (2008) | Google engineering practice (2003) |
| **Focus** | Collaboration, automation, CI/CD | Reliability, measurability, engineering |
| **Team structure** | Embedded in all teams | Dedicated SRE team (or embedded SREs) |
| **Decision framework** | "Ship it faster" | "Ship it if the error budget allows" |
| **Automation goal** | Automate everything | Automate to eliminate toil (< 50% operational work) |
| **Failure approach** | "Move fast, fix forward" | "Move at a sustainable pace, learn from every failure" |

### 1.3 SRE Team Models

| Model | Description | Best For |
|-------|-------------|---------|
| **Centralized SRE** | Dedicated SRE team supports multiple product teams | Large orgs, complex systems |
| **Embedded SRE** | SRE engineer embedded in each product team | Medium orgs, close collaboration |
| **Consulting SRE** | SRE team advises product teams, does not operate services | Growing orgs, knowledge sharing |
| **Everyone does SRE** | No dedicated SRE team; developers own their own reliability | Small orgs, strong engineering culture |

---

## 2. SLOs, SLIs, and SLAs

### 2.1 Definitions

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLI → SLO → SLA                               │
│                                                                  │
│  SLI (Service Level Indicator)                                   │
│  └── A quantitative measure of service behavior                  │
│      Example: "The proportion of requests that complete in       │
│               less than 300ms"                                   │
│                                                                  │
│  SLO (Service Level Objective)                                   │
│  └── A target value for an SLI                                   │
│      Example: "99.9% of requests complete in < 300ms,            │
│               measured over a 30-day rolling window"             │
│                                                                  │
│  SLA (Service Level Agreement)                                   │
│  └── A contract with consequences for missing the SLO            │
│      Example: "If availability drops below 99.9% in a month,    │
│               the customer receives a 10% service credit"        │
│                                                                  │
│  Relationship: SLI measures → SLO sets target → SLA adds teeth  │
│  Rule: SLO should be stricter than SLA (buffer for internal use)│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Choosing SLIs

| Service Type | Recommended SLIs |
|-------------|-----------------|
| **Request-serving (API)** | Availability (% of successful responses), Latency (p50, p95, p99), Error rate |
| **Data processing (pipeline)** | Freshness (time since last successful run), Correctness (% of valid outputs), Coverage (% of data processed) |
| **Storage (database)** | Availability, Durability (% of data not lost), Read/write latency |
| **Streaming (message queue)** | End-to-end latency, Message loss rate, Consumer lag |

### 2.3 Defining SLOs

```yaml
# SLO definition for an API service
service: order-service
slos:
  - name: availability
    description: "Proportion of successful HTTP responses"
    sli:
      type: availability
      good_events: "http_requests_total{status!~'5..'}"
      total_events: "http_requests_total"
    objective: 99.9          # 99.9% availability
    window: 30d              # 30-day rolling window
    # Budget: 0.1% of requests can fail
    # At 10,000 req/min = 10 failed req/min allowed
    # At 10,000 req/min * 30 days = ~432M total requests
    # Error budget = 432,000 allowed failures

  - name: latency
    description: "Proportion of requests faster than 300ms"
    sli:
      type: latency
      good_events: "http_request_duration_seconds_bucket{le='0.3'}"
      total_events: "http_request_duration_seconds_count"
    objective: 99.0          # 99% of requests < 300ms
    window: 30d

  - name: latency-p99
    description: "99th percentile request duration"
    sli:
      type: latency_percentile
      query: "histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))"
    objective_value: 1.0     # p99 < 1 second
    window: 30d
```

### 2.4 The Nines Table

| Availability | Downtime/Year | Downtime/Month | Downtime/Week |
|-------------|--------------|---------------|--------------|
| 99% (two nines) | 3.65 days | 7.3 hours | 1.68 hours |
| 99.9% (three nines) | 8.77 hours | 43.8 minutes | 10.1 minutes |
| 99.95% | 4.38 hours | 21.9 minutes | 5.04 minutes |
| 99.99% (four nines) | 52.6 minutes | 4.38 minutes | 1.01 minutes |
| 99.999% (five nines) | 5.26 minutes | 26.3 seconds | 6.05 seconds |

**Rule of thumb:** Each additional nine costs 10x more to achieve. Going from 99.9% to 99.99% requires fundamentally different architecture (multi-region, automatic failover, no single points of failure).

### 2.5 SLO Implementation with Prometheus

```yaml
# Prometheus recording rules for SLO tracking
groups:
  - name: slo_rules
    interval: 30s
    rules:
      # Total requests in the window
      - record: slo:http_requests:total_rate30d
        expr: sum(rate(http_requests_total{job="order-service"}[30d]))

      # Successful requests in the window
      - record: slo:http_requests:good_rate30d
        expr: sum(rate(http_requests_total{job="order-service",status!~"5.."}[30d]))

      # Current availability (SLI)
      - record: slo:http_requests:availability30d
        expr: slo:http_requests:good_rate30d / slo:http_requests:total_rate30d

      # Error budget remaining (1 = 100% remaining, 0 = exhausted)
      - record: slo:http_requests:error_budget_remaining
        expr: |
          1 - (
            (1 - slo:http_requests:availability30d)
            / (1 - 0.999)
          )

  - name: slo_alerts
    rules:
      # Alert when error budget is below 25%
      - alert: ErrorBudgetLow
        expr: slo:http_requests:error_budget_remaining < 0.25
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Error budget below 25% for order-service"
          description: "Only {{ $value | humanizePercentage }} of error budget remaining."

      # Alert when error budget is exhausted
      - alert: ErrorBudgetExhausted
        expr: slo:http_requests:error_budget_remaining < 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Error budget exhausted for order-service"
          description: "SLO is being violated. Halt feature launches."
```

---

## 3. Error Budgets

### 3.1 What is an Error Budget

An error budget is the inverse of the SLO -- it is the amount of unreliability you are willing to tolerate.

```
SLO: 99.9% availability

Error Budget = 100% - 99.9% = 0.1%

Over 30 days with 1,000,000 requests/day:
Total requests = 30,000,000
Error budget   = 30,000,000 * 0.001 = 30,000 allowed failures

┌────────────────────────────────────────────────────────────┐
│  Error Budget Over 30 Days                                  │
│                                                             │
│  100% ┤████████████████████████████████████████  Budget     │
│       │████████████████████████████████████████  Remaining  │
│  75%  ┤███████████████████████████████████                  │
│       │████████████████████████████████                     │
│  50%  ┤████████████████████████████                         │
│       │██████████████████████████                           │
│  25%  ┤████████████████████████      ← Warning threshold   │
│       │█████████████████████                                │
│  0%   ┤█████████████████             ← Budget exhausted    │
│       └──────────────────────────────────────────────────── │
│        Day 1    Day 10    Day 20    Day 30                  │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Error Budget Policy

An error budget policy defines what happens when the budget is healthy, low, or exhausted:

| Budget Status | Action |
|--------------|--------|
| **> 50% remaining** | Normal operations. Ship features freely. Run chaos experiments. |
| **25-50% remaining** | Caution. Prioritize reliability work. Reduce deployment frequency. Cancel non-essential experiments. |
| **< 25% remaining** | Warning. Freeze feature launches. All engineering effort on reliability. Incident review required. |
| **Exhausted (0%)** | Freeze. No deployments except reliability fixes. Emergency reliability sprint until budget recovers. |

### 3.3 Error Budget Calculation Example

```python
# Error budget tracking example
from datetime import datetime, timedelta

class ErrorBudgetTracker:
    def __init__(self, slo_target: float, window_days: int):
        self.slo_target = slo_target
        self.window_days = window_days
        self.error_budget = 1 - slo_target  # e.g., 0.001 for 99.9%

    def budget_remaining(self, total_requests: int, failed_requests: int) -> float:
        """Returns the fraction of error budget remaining (0.0 to 1.0)."""
        allowed_failures = total_requests * self.error_budget
        actual_failures = failed_requests
        if allowed_failures == 0:
            return 0.0
        return max(0, (allowed_failures - actual_failures) / allowed_failures)

    def burn_rate(self, total_requests: int, failed_requests: int) -> float:
        """
        Returns the burn rate.
        1.0 = consuming budget at exactly the expected rate
        2.0 = consuming budget at 2x the expected rate (will exhaust in half the window)
        """
        actual_error_rate = failed_requests / total_requests if total_requests > 0 else 0
        return actual_error_rate / self.error_budget

# Example:
tracker = ErrorBudgetTracker(slo_target=0.999, window_days=30)
budget = tracker.budget_remaining(total_requests=10_000_000, failed_requests=5_000)
burn = tracker.burn_rate(total_requests=10_000_000, failed_requests=5_000)
print(f"Budget remaining: {budget:.1%}")  # 50.0%
print(f"Burn rate: {burn:.1f}x")          # 5.0x (consuming 5x faster than allowed)
```

### 3.4 Multi-Window Burn Rate Alerts

Google's recommended approach for SLO alerting uses multiple time windows to detect both fast burns (outage) and slow burns (degradation):

```yaml
# Fast burn: 14.4x burn rate over 1 hour (exhausts 30-day budget in 2 days)
- alert: SLOFastBurn
  expr: |
    (
      sum(rate(http_requests_total{job="order-service",status=~"5.."}[1h]))
      / sum(rate(http_requests_total{job="order-service"}[1h]))
    ) > (14.4 * 0.001)
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Fast error budget burn (14.4x) - order-service"

# Slow burn: 3x burn rate over 6 hours (exhausts budget in 10 days)
- alert: SLOSlowBurn
  expr: |
    (
      sum(rate(http_requests_total{job="order-service",status=~"5.."}[6h]))
      / sum(rate(http_requests_total{job="order-service"}[6h]))
    ) > (3 * 0.001)
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Slow error budget burn (3x) - order-service"
```

---

## 4. Toil Elimination

### 4.1 What is Toil

Toil is the kind of work tied to running a production service that is manual, repetitive, automatable, tactical, devoid of enduring value, and scales linearly with service growth.

| Characteristic | Toil | Not Toil |
|---------------|------|----------|
| **Manual** | Human runs a script or clicks buttons | Automated pipeline |
| **Repetitive** | Same task every week/day/hour | One-time project |
| **Automatable** | Could be done by software | Requires human judgment |
| **Tactical** | Interrupt-driven, reactive | Planned, strategic |
| **No lasting value** | System returns to previous state | Permanent improvement |
| **Scales with growth** | More users = more toil | System handles growth automatically |

### 4.2 Examples

| Toil | Engineering Solution |
|------|---------------------|
| Manually restarting crashed pods | Configure liveness probes and automatic restarts |
| Manually scaling during traffic spikes | Implement Horizontal Pod Autoscaler |
| Manually rotating TLS certificates | Deploy cert-manager with auto-renewal |
| Manually reviewing and approving routine deploys | Implement automated canary analysis |
| Running weekly database maintenance queries | Schedule CronJobs with automated validation |
| Manually provisioning user accounts | Build self-service user provisioning |

### 4.3 The 50% Rule

Google's SRE principle: **an SRE team should spend no more than 50% of its time on toil.** The other 50% must be spent on engineering work that permanently reduces toil or improves reliability.

```
Healthy SRE Team:                  Unhealthy SRE Team:
┌─────────────────────┐           ┌─────────────────────┐
│  Engineering  50%   │           │  Engineering  20%   │
│  ████████████████   │           │  ██████             │
│                     │           │                     │
│  Toil         50%   │           │  Toil         80%   │
│  ████████████████   │           │  ████████████████   │
│                     │           │  ████████████████   │
│                     │           │  ████████████████   │
└─────────────────────┘           └─────────────────────┘
 Result: Toil decreases            Result: Team burns out,
 over time as automation            never improves, sinks
 replaces manual work               deeper into toil
```

### 4.4 Toil Tracking

```yaml
# Toil budget tracking (quarterly review)
team: sre-team
quarter: Q1-2024
team_size: 5
total_hours: 5 * 40 * 13  # 2,600 hours

toil_budget: 1300  # 50% max

toil_items:
  - name: "Manual certificate rotation"
    hours_per_quarter: 40
    frequency: "Monthly per service (20 services)"
    automation_effort: "80 hours one-time"
    priority: high

  - name: "Incident triage and paging"
    hours_per_quarter: 260
    frequency: "On-call rotation"
    automation_effort: "Improve alerting to reduce false positives (120 hours)"
    priority: high

  - name: "Capacity planning reviews"
    hours_per_quarter: 80
    frequency: "Weekly"
    automation_effort: "Auto-scaling + predictive alerts (200 hours)"
    priority: medium

  - name: "Database schema migration assistance"
    hours_per_quarter: 60
    frequency: "Per-request"
    automation_effort: "Self-service migration tool (160 hours)"
    priority: low

total_toil: 440  # 17% of total hours (healthy)
```

---

## 5. Incident Management

### 5.1 Incident Lifecycle

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Detect  │──→│  Triage  │──→│ Mitigate │──→│ Resolve  │──→│  Review  │
│          │   │          │   │          │   │          │   │(Postmort)│
│ Alert    │   │ Severity │   │ Stop the │   │ Fix root │   │ Learn &  │
│ fires    │   │ assign   │   │ bleeding │   │ cause    │   │ prevent  │
│          │   │ IC named │   │          │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
  Minutes        Minutes       Minutes-Hours    Hours-Days      Days
```

### 5.2 Severity Levels

| Severity | Impact | Response Time | Examples |
|----------|--------|--------------|---------|
| **SEV-1 (Critical)** | Full service outage, data loss, security breach | Immediate (< 5 min) | Site is down, database corruption, active breach |
| **SEV-2 (Major)** | Significant feature degradation, partial outage | < 15 minutes | Payment processing down, 50% error rate |
| **SEV-3 (Minor)** | Minor feature impact, performance degradation | < 1 hour | Slow responses, non-critical feature broken |
| **SEV-4 (Low)** | No user impact, internal tooling issue | Next business day | Internal dashboard down, dev environment issue |

### 5.3 Incident Roles

| Role | Responsibility |
|------|----------------|
| **Incident Commander (IC)** | Coordinates the response. Makes decisions. Delegates tasks. Does NOT debug. |
| **Communications Lead** | Updates stakeholders, status page, and Slack channel. Manages external comms. |
| **Operations Lead** | Executes technical mitigation (restarts, rollbacks, scaling). |
| **Subject Matter Expert (SME)** | Deep knowledge of affected system. Investigates root cause. |
| **Scribe** | Records timeline, actions taken, and decisions in the incident document. |

### 5.4 Incident Communication Template

```markdown
# Incident: [Title]
**Severity**: SEV-[1/2/3]
**Status**: [Investigating / Identified / Monitoring / Resolved]
**Incident Commander**: [Name]
**Start Time**: [ISO 8601]

## Impact
[What is broken? How many users are affected?]

## Timeline
| Time (UTC) | Event |
|-----------|-------|
| 14:23 | Alert: HighErrorRate fired for order-service |
| 14:25 | IC assigned: [Name] |
| 14:28 | Identified: Database connection pool exhausted |
| 14:32 | Mitigation: Restarted order-service pods |
| 14:35 | Monitoring: Error rate dropping |
| 14:45 | Resolved: Error rate back to baseline |

## Root Cause
[Brief description]

## Mitigation Applied
[What was done to stop the bleeding]

## Action Items
- [ ] [Action 1] — Owner: [Name] — Due: [Date]
- [ ] [Action 2] — Owner: [Name] — Due: [Date]
```

### 5.5 Incident Response Tooling

| Tool | Purpose |
|------|---------|
| **PagerDuty / OpsGenie** | Alerting, on-call management, escalation |
| **Slack / Teams** | Real-time incident communication channel |
| **Statuspage** | External communication (customer-facing status) |
| **Jira / Linear** | Track post-incident action items |
| **Grafana / Datadog** | Real-time metric visualization during incidents |
| **Rootly / Incident.io** | Automated incident management workflow |

---

## 6. Postmortems

### 6.1 Blameless Postmortems

A blameless postmortem focuses on what happened and how to prevent it, not who caused it. The premise is that people make rational decisions based on the information available to them at the time. If someone deployed a bad change, the system should have caught it -- not the person.

**Key principles:**
- Focus on **systems** and **processes**, not **people**
- Assume good intent from everyone involved
- Ask "What caused the system to fail?" not "Who caused the failure?"
- The goal is to make the system safer, not to assign blame

### 6.2 Postmortem Template

```markdown
# Postmortem: [Incident Title]

**Date**: [Date of incident]
**Duration**: [Start time — End time]
**Severity**: SEV-[1/2/3]
**Authors**: [Names]
**Status**: [Draft / Final]

## Summary
[2-3 sentence summary of what happened and the impact]

## Impact
- **Duration**: X hours Y minutes
- **Users affected**: N users / N% of traffic
- **Revenue impact**: $X (if applicable)
- **Data lost**: [None / Description]

## Root Cause
[Technical description of what broke and why]

## Trigger
[What specific event initiated the incident?
 e.g., "A configuration change at 14:15 UTC increased the connection pool timeout from 5s to 60s"]

## Detection
- **How detected**: [Alert / Customer report / Manual observation]
- **Time to detect**: [Minutes from trigger to first alert]
- **Detection gap**: [Was there a monitoring gap? Should an alert have fired earlier?]

## Timeline
| Time (UTC) | Event |
|-----------|-------|
| 14:15 | Config change deployed via CI/CD |
| 14:23 | Error rate exceeds 5% — HighErrorRate alert fires |
| 14:25 | IC assigned. War room opened in Slack #inc-20240315 |
| 14:28 | SME identifies connection pool exhaustion in logs |
| 14:32 | Mitigation: rollback config change |
| 14:35 | Error rate declining |
| 14:45 | Error rate back to baseline. Incident resolved. |

## Root Cause Analysis
[Detailed technical analysis using the "5 Whys" or fishbone diagram]

**5 Whys:**
1. Why did orders fail? → Database connections timed out
2. Why did connections time out? → Connection pool was exhausted
3. Why was the pool exhausted? → Timeout was set to 60s (should be 5s)
4. Why was the timeout changed? → Config change in PR #1234
5. Why was the bad config merged? → No validation or staging test for timeout values

## What Went Well
- Alert fired within 8 minutes of the trigger
- IC was assigned within 2 minutes of the alert
- Rollback was clean and fast (3 minutes)

## What Went Wrong
- Config change was not tested in staging before production
- No automated validation for connection pool timeout values
- Runbook for "connection pool exhaustion" did not exist

## Where We Got Lucky
- The incident happened during business hours with the full team available
- The config change was easily reversible

## Action Items
| # | Action | Type | Owner | Due Date | Status |
|---|--------|------|-------|----------|--------|
| 1 | Add config validation for timeout values (min: 1s, max: 30s) | Prevention | @alice | 2024-03-22 | TODO |
| 2 | Require staging deployment before production for config changes | Prevention | @bob | 2024-03-29 | TODO |
| 3 | Create runbook for connection pool exhaustion | Mitigation | @carol | 2024-03-20 | TODO |
| 4 | Add alert for connection pool utilization > 80% | Detection | @dave | 2024-03-18 | DONE |

## Lessons Learned
[What did we learn that applies beyond this specific incident?]
```

### 6.3 Postmortem Review Process

1. **Write**: IC and SMEs draft the postmortem within 48 hours
2. **Review**: Team reviews for accuracy and completeness
3. **Share**: Published to the engineering organization (transparency)
4. **Track**: Action items are tracked in the issue tracker
5. **Follow up**: Review action item completion after 2 weeks

---

## 7. On-Call Practices

### 7.1 Sustainable On-Call

| Practice | Description |
|----------|-------------|
| **Rotation size** | Minimum 8 people per rotation (ensures no one is on-call more than every 8 weeks) |
| **Shift length** | 1 week primary, 1 week secondary. No back-to-back weeks. |
| **Compensation** | On-call pay or compensatory time off |
| **Page volume** | Target < 2 pages per on-call shift. More than 5 requires action. |
| **Handoff** | Written handoff document at rotation change (active incidents, known risks) |
| **Escalation** | Clear escalation path: primary → secondary → team lead → management |
| **Follow-the-sun** | For global teams, hand off to the next time zone instead of overnight pages |

### 7.2 On-Call Health Metrics

| Metric | Healthy | Needs Attention | Unsustainable |
|--------|---------|----------------|--------------|
| **Pages per week** | 0-2 | 3-5 | > 5 |
| **Off-hours pages** | 0-1 | 2-3 | > 3 |
| **MTTA (Mean Time to Acknowledge)** | < 5 min | 5-15 min | > 15 min |
| **False positive rate** | < 10% | 10-30% | > 30% |
| **Toil percentage** | < 30% | 30-50% | > 50% |
| **Sleep interruptions** | 0-1 per week | 2-3 per week | > 3 per week |

### 7.3 Reducing On-Call Burden

```
┌──────────────────────────────────────────────────────────────┐
│              On-Call Improvement Flywheel                      │
│                                                               │
│  1. Measure page volume and false positives                   │
│       │                                                       │
│       ▼                                                       │
│  2. Fix top sources of pages                                  │
│     ├── Tune noisy alerts (reduce false positives)            │
│     ├── Auto-remediate common issues (self-healing)           │
│     └── Fix root causes (not just symptoms)                   │
│       │                                                       │
│       ▼                                                       │
│  3. Fewer pages → better sleep → better engineering           │
│       │                                                       │
│       ▼                                                       │
│  4. Better engineering → more automation → fewer pages        │
│       │                                                       │
│       └──────────────→ (repeat)                               │
└──────────────────────────────────────────────────────────────┘
```

### 7.4 On-Call Handoff Template

```markdown
# On-Call Handoff: [Date]

## Outgoing: [Name]
## Incoming: [Name]

## Active Issues
- [ ] order-service showing elevated p99 latency since Tuesday.
      Likely related to DB index rebuild. Monitor #alert-backend.

## Recent Incidents
- SEV-2 on Monday: Payment timeout. Root cause fixed. Postmortem in progress.

## Upcoming Risks
- Database migration scheduled for Thursday 2 AM UTC.
  Runbook: [link]. Rollback plan: [link].

## Known Alert Noise
- `DiskSpaceLow` on node-7 is a false positive (large log file, rotation scheduled).
  Silence ID: abc123 (expires Friday).

## Tips
- If order-service pages, check the Redis cluster first (it was flaky last week).
- The staging environment is broken; do not deploy there until PR #456 is merged.
```

---

## 8. SRE Decision Framework

### 8.1 Using Error Budgets for Decisions

```
Decision: "Should we deploy Feature X to production?"

┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Step 1: Check error budget                              │
│  ├── Budget > 50%  →  Yes, deploy freely                │
│  ├── Budget 25-50% →  Deploy with extra monitoring      │
│  ├── Budget < 25%  →  Deploy only if it improves        │
│  │                    reliability                        │
│  └── Budget = 0%   →  No. Only reliability fixes.       │
│                                                          │
│  Step 2: Check deployment risk                           │
│  ├── Low risk (config change, minor UI)  → Rolling      │
│  ├── Medium risk (new feature)           → Canary       │
│  └── High risk (database migration)      → Blue-green   │
│                                                          │
│  Step 3: Monitor after deployment                        │
│  ├── Error budget burn rate < 1x → Healthy              │
│  ├── Error budget burn rate 1-3x → Monitor closely      │
│  └── Error budget burn rate > 3x → Rollback             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 8.2 Reliability vs Velocity Trade-off

```
                    Error Budget Remaining
  100% ┤═══════════════════════════════════════════
       │        "Ship features freely"
       │
  75%  ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
       │        "Normal velocity"
       │
  50%  ┤═══════════════════════════════════════════
       │        "Slow down, prioritize reliability"
       │
  25%  ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
       │        "Feature freeze, reliability sprint"
       │
   0%  ┤═══════════════════════════════════════════
       │        "SLO violated, freeze all deploys"
       └────────────────────────────────────────────
```

---

## 9. Next Steps

- [10_Monitoring_and_Alerting.md](./10_Monitoring_and_Alerting.md) - Monitoring fundamentals that underpin SRE
- [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) - Proactive reliability testing

---

## Exercises

### Exercise 1: SLO Design

Design SLOs for the following services:

1. An e-commerce checkout API that processes 50,000 requests per day
2. A data pipeline that processes clickstream events from a mobile app
3. An internal admin dashboard used by 20 employees during business hours

<details>
<summary>Show Answer</summary>

**1. E-commerce Checkout API:**

| SLO | SLI | Target | Window | Rationale |
|-----|-----|--------|--------|-----------|
| Availability | `successful_responses / total_responses` | 99.95% | 30 days | Payment-critical; 99.9% allows ~50 failures/day, too many for checkout. 99.95% allows ~25/day. |
| Latency (fast) | `requests < 500ms / total_requests` | 99% | 30 days | Most checkouts should be fast; slow checkouts cause cart abandonment. |
| Latency (ceiling) | p99 latency | < 2s | 30 days | Even the slowest requests should complete before user patience expires. |

**Error budget**: 99.95% over 30 days with 50,000 req/day = 1,500,000 total requests. Budget = 750 allowed failures per month (25/day).

**2. Data Pipeline:**

| SLO | SLI | Target | Window | Rationale |
|-----|-----|--------|--------|-----------|
| Freshness | `time_since_last_successful_run` | < 15 minutes | Rolling | Clickstream data is used for near-real-time personalization. |
| Correctness | `valid_records_output / total_records_input` | 99.9% | 30 days | Dropping < 0.1% of events is acceptable for analytics. |
| Coverage | `records_processed / records_received` | 99.99% | 30 days | Almost no data should be silently dropped. |

**3. Internal Admin Dashboard:**

| SLO | SLI | Target | Window | Rationale |
|-----|-----|--------|--------|-----------|
| Availability | `successful_responses / total_responses` | 99% (during business hours) | 30 days, business hours only | Internal tool with 20 users; occasional downtime is tolerable. |
| Latency | p95 page load time | < 3s | 30 days | Internal users are more tolerant of slower pages. |

**Key insight**: The checkout API (99.95%) has a much stricter SLO than the admin dashboard (99%) because the business impact of failure is orders of magnitude different. SLOs should reflect business impact, not technical capability.

</details>

### Exercise 2: Error Budget Scenario

Your order-service has an SLO of 99.9% availability over a 30-day rolling window. It serves 100,000 requests per day.

On Day 15, a bad deployment causes a 2-hour outage during which 80% of requests fail. Calculate:
1. The total error budget for the 30-day window
2. How much budget was consumed by the outage
3. The remaining budget after the outage
4. What actions should be taken based on the error budget policy

<details>
<summary>Show Answer</summary>

**1. Total error budget:**
```
Total requests in 30 days = 100,000/day * 30 days = 3,000,000
Error budget = 3,000,000 * (1 - 0.999) = 3,000 allowed failures
```

**2. Budget consumed by the outage:**
```
Outage duration: 2 hours
Requests during outage: 100,000/day / 24 hours * 2 hours = 8,333 requests
Failed requests: 8,333 * 80% = 6,667 failures
```

**3. Remaining budget:**
```
Assuming no other failures in the 30-day window:
Budget consumed: 6,667 failures
Budget remaining: 3,000 - 6,667 = -3,667

The error budget is EXHAUSTED (negative).
The team has exceeded their SLO by 3,667 failures.
```

Wait -- the budget consumed (6,667) exceeds the entire 30-day budget (3,000). This means:

```
Budget remaining as percentage: (3,000 - 6,667) / 3,000 = -122%
The SLO is violated.
```

**4. Actions based on error budget policy:**

Since the budget is exhausted (and significantly so), the following actions apply:

1. **Immediate feature freeze**: No new feature deployments until the error budget recovers. Only reliability fixes are deployed.

2. **Mandatory postmortem**: Conduct a blameless postmortem within 48 hours covering:
   - Why the bad deployment was not caught in staging
   - Why the canary/rollback did not prevent 2 hours of impact
   - Why monitoring did not detect the issue sooner

3. **Reliability sprint**: Allocate the next sprint entirely to reliability improvements:
   - Implement automated canary analysis (if not already in place)
   - Add deployment validation tests
   - Improve rollback speed

4. **Recovery timeline**: The 30-day rolling window means the budget will "recover" as the outage day (Day 15) rolls out of the window on Day 45. However, the team should not simply wait -- they must demonstrate reliability improvements before resuming normal deployment cadence.

5. **Communication**: Report the SLO violation to stakeholders. If there is an external SLA, check whether the SLA was also violated and what contractual obligations apply.

</details>

### Exercise 3: Toil Classification

Classify each of the following tasks as **toil** or **not toil**, and explain your reasoning:

1. Running a database backup script every night at 2 AM
2. Writing a design document for a new caching layer
3. Restarting a service that crashes every 3 days due to a memory leak
4. Reviewing and merging pull requests from team members
5. Manually adding new users to the monitoring system when they join the company
6. Investigating a novel production incident that has never happened before

<details>
<summary>Show Answer</summary>

1. **Running a database backup script every night at 2 AM -- TOIL**
   - Manual: Someone must run or trigger the script.
   - Repetitive: Every night.
   - Automatable: A CronJob or managed backup service can do this.
   - No lasting value: Each backup is consumed and replaced.
   - **Fix**: Use AWS RDS automated backups, or a Kubernetes CronJob with validation.

2. **Writing a design document for a new caching layer -- NOT TOIL**
   - Requires human judgment and creativity.
   - One-time task with lasting value (the design informs future work).
   - This is engineering work.

3. **Restarting a service that crashes every 3 days due to a memory leak -- TOIL**
   - Manual and repetitive.
   - Automatable (but should not just be automated -- the memory leak should be fixed).
   - No lasting value (service crashes again in 3 days).
   - **Fix**: Fix the memory leak (root cause). As a temporary measure, add a liveness probe with memory-based restart, but prioritize the permanent fix.

4. **Reviewing and merging pull requests -- NOT TOIL**
   - Requires human judgment (code quality, correctness, design).
   - Not automatable in a meaningful way (linting is, but review is not).
   - Provides lasting value (knowledge sharing, quality enforcement).

5. **Manually adding new users to the monitoring system -- TOIL**
   - Manual, repetitive, automatable.
   - Scales linearly with company growth.
   - No lasting value (same process every time).
   - **Fix**: Integrate monitoring with the identity provider (SSO/LDAP). New users are auto-provisioned.

6. **Investigating a novel production incident -- NOT TOIL**
   - Not repetitive (novel incident by definition).
   - Requires human judgment and deep technical analysis.
   - Provides lasting value (root cause understanding, postmortem, action items).
   - This is exactly the kind of work SREs should be doing.

</details>

### Exercise 4: Postmortem Writing

An incident occurred with the following facts:

- **Service**: payment-service
- **Duration**: 45 minutes (09:15 - 10:00 UTC)
- **Impact**: 100% of credit card payments failed; debit card payments were unaffected
- **Root cause**: A third-party payment processor (Stripe) deployed a breaking API change to their `/v1/charges` endpoint. The response format changed from `{"id": "ch_xxx"}` to `{"charge": {"id": "ch_xxx"}}`.
- **Detection**: Customer support received 15 complaints before an alert fired. The alert fired 12 minutes after the first failures.
- **Mitigation**: Hotfix deployed to parse both response formats.

Write the action items section of the postmortem with at least 5 items, categorized by prevention, detection, and mitigation.

<details>
<summary>Show Answer</summary>

**Action Items:**

| # | Action | Category | Owner | Due Date | Priority |
|---|--------|----------|-------|----------|----------|
| 1 | **Add response schema validation to Stripe API client**: Validate the structure of Stripe responses before parsing. If the schema is unexpected, log a structured error and trigger an alert before processing fails. | Prevention | @payments-team | 2024-04-01 | P1 |
| 2 | **Subscribe to Stripe API changelog and deprecation notices**: Set up automated monitoring of Stripe's API changelog RSS feed. Alert the payments team when breaking changes are announced. | Prevention | @payments-team | 2024-03-25 | P2 |
| 3 | **Pin Stripe API version in request headers**: Use `Stripe-Version: 2024-02-01` header to lock our integration to a known API version, preventing surprise changes from affecting us. | Prevention | @payments-team | 2024-03-22 | P1 |
| 4 | **Fix alert gap: add payment failure rate alert**: Create a Prometheus alert that fires when the credit card payment failure rate exceeds 5% for 2 consecutive minutes. Current alert threshold was too high (was 20%), which delayed detection by 12 minutes. | Detection | @sre-team | 2024-03-20 | P0 |
| 5 | **Add synthetic payment monitoring**: Deploy a synthetic test that executes a $0.50 test charge every 5 minutes. Alert immediately if the test charge fails. This would have detected the issue before real customers were affected. | Detection | @sre-team | 2024-04-05 | P1 |
| 6 | **Create runbook for "payment processor API failure"**: Document the steps to diagnose and mitigate failures in external payment APIs, including: how to identify which payment methods are affected, how to route traffic to a backup processor, and how to deploy a hotfix. | Mitigation | @payments-team | 2024-03-25 | P2 |
| 7 | **Implement payment processor fallback**: Evaluate adding a second payment processor (e.g., Adyen) as a fallback. If Stripe returns errors for > 1 minute, automatically route traffic to the backup processor. | Mitigation | @payments-team | 2024-05-01 | P2 |

**Key lessons:**
- The 12-minute detection gap (alert fired after 15 customer complaints) is the most critical item to fix (Action 4).
- Pinning the API version (Action 3) would have prevented this specific incident entirely.
- The synthetic test (Action 5) provides the earliest possible detection, independent of real user traffic.

</details>

---

## References

- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [Google SRE Workbook](https://sre.google/workbook/table-of-contents/)
- [The Art of SLOs (Google)](https://sre.google/resources/practices-and-processes/art-of-slos/)
- [Implementing SLOs (O'Reilly)](https://www.oreilly.com/library/view/implementing-service-level/9781492076803/)
- [Incident Management Guide (PagerDuty)](https://response.pagerduty.com/)
- [Blameless Postmortems (Etsy)](https://www.etsy.com/codeascraft/blameless-postmortems/)
