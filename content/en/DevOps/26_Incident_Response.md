# 26. Incident Response

**Previous**: [Continuous Profiling](./25_Continuous_Profiling.md) | **Next**: [AIOps and Anomaly Detection](./27_AIOps_Anomaly_Detection.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design and operate a structured incident management process with clear roles and escalation paths
2. Implement sustainable on-call practices that balance team well-being with system reliability
3. Conduct blameless postmortems that produce actionable improvements
4. Write effective runbooks that accelerate incident mitigation
5. Measure incident response effectiveness using MTTR, MTTD, and incident severity classifications
6. Build an incident response culture that treats failures as learning opportunities

---

Incidents are inevitable in complex systems. The question is not whether they will happen, but how effectively your team responds when they do. A well-structured incident response process reduces the blast radius, shortens resolution time, and -- most importantly -- generates the learning that prevents recurrence.

> **Analogy -- Hospital Emergency Response**: When a patient arrives at the ER, there is no ad-hoc scramble. There is a structured process: triage (severity assessment), a trauma team with assigned roles (lead physician, nurse, anesthesiologist), clear communication protocols (SBAR), and after the case -- a morbidity and mortality review (postmortem) to improve future care. Software incident response follows the same principles.

## 1. Incident Severity Classification

### 1.1 Severity Levels

| Level | Name | Criteria | Response | Example |
|-------|------|----------|----------|---------|
| **SEV1** | Critical | Revenue-impacting outage affecting all users | Page on-call, assemble incident team, exec communication | Payment processing completely down |
| **SEV2** | Major | Significant degradation affecting many users | Page on-call, incident commander leads response | 50% of API requests returning 500s |
| **SEV3** | Minor | Partial degradation, workaround available | Alert on-call, investigate during business hours | Search results slow but functional |
| **SEV4** | Low | Cosmetic or minor issue, no user impact | Create ticket, address in next sprint | Dashboard shows stale data |

### 1.2 Severity Decision Tree

```
Is the issue causing data loss or security breach?
├── YES → SEV1 (always)
└── NO
    Is revenue or core functionality affected?
    ├── YES
    │   How many users affected?
    │   ├── > 50% → SEV1
    │   ├── 10-50% → SEV2
    │   └── < 10% → SEV3
    └── NO
        Is a workaround available?
        ├── NO → SEV3
        └── YES → SEV4
```

---

## 2. Incident Response Process

### 2.1 The Incident Lifecycle

```
Detection → Triage → Response → Mitigation → Resolution → Postmortem
   │          │         │           │             │            │
   │          │         │           │             │            └── Learn and improve
   │          │         │           │             └── Root cause fixed permanently
   │          │         │           └── Bleeding stopped (temporary fix)
   │          │         └── Team assembled, working on fix
   │          └── Severity assessed, roles assigned
   └── Alert fires or user reports issue
```

### 2.2 Incident Roles

| Role | Responsibility | Who |
|------|---------------|-----|
| **Incident Commander (IC)** | Coordinates response, makes decisions, manages communication | Senior engineer or on-call lead |
| **Technical Lead** | Drives technical investigation and fix | Subject matter expert |
| **Communication Lead** | Updates stakeholders, customers, status page | IC or designated person |
| **Scribe** | Documents timeline, actions, findings in real-time | Any team member |

### 2.3 Incident Communication Template

```markdown
## Incident: [TITLE]
**Severity**: SEV2
**Status**: Investigating / Identified / Monitoring / Resolved
**Incident Commander**: @alice
**Tech Lead**: @bob

### Timeline (UTC)
- 14:00 - Alert fired: payment-service error rate > 5%
- 14:03 - IC acknowledged, beginning investigation
- 14:08 - Identified: Stripe API timeout causing cascading failures
- 14:12 - Mitigation: Enabled fallback payment processor
- 14:15 - Error rate returning to normal
- 14:30 - Monitoring: All metrics within SLO
- 15:00 - Resolved: Stripe API recovered, reverted to primary processor

### Impact
- Duration: 30 minutes (14:00 - 14:30)
- Users affected: ~5,000 payment attempts failed
- Revenue impact: ~$50,000 in delayed transactions (all eventually processed)

### Root Cause
Stripe API experienced elevated latency (>30s) due to their internal infrastructure issue.
Our 30-second timeout caused all in-flight requests to fail.

### Action Items
- [ ] P1: Reduce Stripe timeout to 5s with circuit breaker (prevent cascade)
- [ ] P1: Enable automatic failover to backup payment processor
- [ ] P2: Add Stripe API latency to our SLO dashboard
```

---

## 3. On-Call Practices

### 3.1 Sustainable On-Call Design

| Principle | Implementation |
|-----------|---------------|
| **Minimum team size: 8** | Ensures no one is on-call more than every 8 weeks |
| **Rotation period: 1 week** | Long enough to build context, short enough to prevent burnout |
| **Compensation** | Extra pay, comp days, or reduced sprint load during on-call week |
| **Primary + Secondary** | Secondary is backup; primary handles first response |
| **Follow-the-sun** | For global teams: hand off on-call between time zones |
| **No heroes** | If one person gets paged > 3x/week, the system needs fixing, not the person |

### 3.2 On-Call Handoff Process

```
Outgoing on-call engineer:
  1. Write handoff notes:
     - Active incidents or ongoing issues
     - Recent deployments that might cause problems
     - Known flaky alerts (with ticket references for fixing them)
     - Anything unusual observed during the shift
  2. Ensure monitoring is green (or document known yellows/reds)
  3. Brief the incoming engineer in a 15-minute sync

Incoming on-call engineer:
  1. Verify pager access (test page yourself)
  2. Review open incident tickets
  3. Review recent deployments (last 48 hours)
  4. Confirm access to all critical dashboards and runbooks
  5. Acknowledge handoff in Slack/channel
```

### 3.3 Alert Hygiene

| Problem | Impact | Fix |
|---------|--------|-----|
| **Alert fatigue** | Engineers ignore all alerts | Reduce to SLO-based alerts (Lesson 20) |
| **Noisy alerts** | Pages for non-actionable events | Require action for every alert; delete or tune otherwise |
| **Missing runbooks** | Engineer spends 20 min figuring out what to do | Every alert MUST link to a runbook |
| **Duplicate alerts** | Same incident triggers 5 pages | Configure alert grouping and inhibition in Alertmanager |
| **Off-hours pages for SEV3** | Sleep disruption for non-urgent issues | Route SEV3/4 to ticket queue, not pager |

### 3.4 Alert-to-Runbook Linking

```yaml
# Prometheus alert with runbook link
- alert: PaymentServiceHighErrorRate
  expr: payment_service:error_ratio:rate5m > 0.01
  for: 3m
  labels:
    severity: critical
    team: payments
    runbook: "https://wiki.example.com/runbooks/payment-high-error-rate"
  annotations:
    summary: "Payment service error rate is {{ $value | humanizePercentage }}"
    dashboard: "https://grafana.example.com/d/payment-slo"
    description: |
      Error rate exceeds 1% for 3+ minutes.
      Check Stripe API status and database connectivity.
```

---

## 4. Runbooks

### 4.1 Runbook Structure

```markdown
# Runbook: Payment Service High Error Rate

## Overview
This runbook addresses alerts for elevated error rates in the payment service.
**Alert**: PaymentServiceHighErrorRate
**Severity**: Critical (pages on-call)
**Service**: payment-service
**Dashboard**: [Payment SLO Dashboard](https://grafana.example.com/d/payment-slo)

## Diagnostic Steps

### Step 1: Identify the error type
```promql
# Check error breakdown
sum by (status) (rate(http_requests_total{job="payment-service",status=~"5.."}[5m]))
```
- If mostly 502/504 → upstream dependency issue (Step 2)
- If mostly 500 → internal error (Step 3)
- If mostly 503 → service overloaded (Step 4)

### Step 2: Check upstream dependencies
1. Check [Stripe Status Page](https://status.stripe.com)
2. Check database connectivity:
   ```bash
   kubectl exec -it deploy/payment-service -- pg_isready -h postgres
   ```
3. Check [dependency dashboard](https://grafana.example.com/d/deps)

### Step 3: Check application health
```bash
kubectl logs -l app=payment-service --tail=100 | grep ERROR
kubectl top pods -l app=payment-service
```

### Step 4: Service overload mitigation
```bash
# Scale up
kubectl scale deploy/payment-service --replicas=10

# If scale-up doesn't help, enable rate limiting
kubectl set env deploy/payment-service RATE_LIMIT_RPS=100
```

## Mitigation Actions
- **Upstream dependency down**: Enable fallback payment processor
  ```bash
  kubectl set env deploy/payment-service PAYMENT_FALLBACK=true
  ```
- **Database issue**: Failover to read replica for non-write operations
- **Application crash loop**: Rollback to last known good version
  ```bash
  kubectl rollout undo deploy/payment-service
  ```

## Escalation
- After 15 minutes without resolution: page payments-team lead
- After 30 minutes: page engineering director
- Revenue impact > $100K: page VP Engineering
```

### 4.2 Runbook Testing

```bash
# Regularly verify runbook steps work (game day exercise)
# Schedule monthly runbook review:
# 1. Run through each diagnostic step -- do the commands work?
# 2. Verify dashboard links are not broken
# 3. Verify escalation contacts are current
# 4. Check if any new failure modes need to be added
```

---

## 5. Blameless Postmortems

### 5.1 Postmortem Principles

| Principle | Implementation |
|-----------|---------------|
| **Blameless** | Focus on systems and processes, not individuals |
| **Timely** | Conduct within 48 hours while memory is fresh |
| **Thorough** | Include timeline, root cause, contributing factors, and action items |
| **Action-oriented** | Every postmortem produces concrete, assigned, and tracked action items |
| **Shared** | Published to the entire engineering org for learning |

### 5.2 Postmortem Template

```markdown
# Postmortem: Payment Processing Outage (2025-03-10)

## Summary
Payment processing was completely unavailable for 45 minutes (09:15-10:00 UTC)
due to a Stripe API breaking change. 100% of credit card payments failed;
debit card payments were unaffected.

## Impact
- **Duration**: 45 minutes
- **User impact**: ~12,000 failed payment attempts
- **Revenue impact**: ~$180,000 in delayed transactions
- **SLO impact**: Error budget consumed from 65% to 12%

## Timeline (UTC)
| Time | Event |
|------|-------|
| 09:10 | Stripe deploys API change to `/v1/charges` endpoint |
| 09:15 | First payment failure logged |
| 09:22 | Alert fires: payment error rate > 1% |
| 09:25 | On-call engineer acknowledges, begins investigation |
| 09:30 | IC assigned (@alice), SEV1 declared |
| 09:35 | Root cause identified: Stripe response format changed |
| 09:42 | Hotfix PR opened to handle both response formats |
| 09:48 | Hotfix deployed to production via emergency pipeline |
| 09:52 | Error rate dropping, monitoring recovery |
| 10:00 | Error rate back to baseline, incident resolved |

## Root Cause
Stripe deployed a breaking change to their `/v1/charges` API endpoint.
The response format changed from `{"id": "ch_xxx"}` to
`{"charge": {"id": "ch_xxx"}}`. Our Stripe client library parsed
the response expecting the old format and threw a deserialization error
for every request.

## Contributing Factors
1. We were not pinning the Stripe API version in request headers
2. Our Stripe client did not validate response schema before parsing
3. Alert threshold was set too high (1%), missing the initial 0.5% spike at 09:15
4. No synthetic payment monitoring to detect failures before real users

## What Went Well
- Hotfix was deployed in 13 minutes after root cause identification
- Communication was clear and timely (status page updated at 09:32)
- Incident commander kept the response focused

## What Went Poorly
- 7-minute detection gap (09:15 to 09:22)
- No automated failover to backup payment processor
- Stripe changelog not monitored for breaking changes

## Action Items
| # | Action | Category | Owner | Priority | Due |
|---|--------|----------|-------|----------|-----|
| 1 | Pin Stripe API version in headers | Prevention | @bob | P1 | 2025-03-14 |
| 2 | Add response schema validation | Prevention | @bob | P1 | 2025-03-17 |
| 3 | Lower alert threshold to 0.3% | Detection | @alice | P0 | 2025-03-11 |
| 4 | Add synthetic payment monitoring | Detection | @carol | P1 | 2025-03-21 |
| 5 | Implement auto-failover to backup processor | Mitigation | @dave | P2 | 2025-04-01 |
| 6 | Subscribe to Stripe API changelog | Prevention | @bob | P2 | 2025-03-14 |

## Lessons Learned
1. External API dependencies should always pin a specific version
2. Response validation catches breaking changes before they cascade
3. Synthetic monitoring detects issues before real users are affected
```

### 5.3 Postmortem Review Meeting

```
Agenda (45-60 minutes):
1. (5 min)  IC reads the summary and timeline aloud
2. (10 min) Walk through the timeline -- anyone can add missing details
3. (15 min) Discuss root cause and contributing factors
4. (10 min) Review "what went well" and "what went poorly"
5. (10 min) Review and prioritize action items
6. (5 min)  Assign owners and due dates

Ground rules:
- No blame: "The system allowed X to happen" not "Person Y caused X"
- Everyone involved is invited; attendance is encouraged, not mandatory
- Focus on systems improvement, not individual performance
- Action items must be specific, assigned, and tracked in the issue tracker
```

---

## 6. Incident Metrics

### 6.1 Key Incident Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **MTTD** (Mean Time to Detect) | Time from incident start to alert firing | < 5 minutes |
| **MTTA** (Mean Time to Acknowledge) | Time from alert to human acknowledgment | < 5 minutes |
| **MTTR** (Mean Time to Resolve) | Time from detection to resolution | < 1 hour (SEV1) |
| **MTBF** (Mean Time Between Failures) | Time between incidents of the same type | Increasing trend |
| **Incident count** | Number of incidents per period | Decreasing trend |
| **Action item completion rate** | Percentage of postmortem action items completed on time | > 90% |

### 6.2 Tracking and Reporting

```python
"""Incident metrics dashboard data."""
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class IncidentMetrics:
    severity: str
    detected_at: str
    acknowledged_at: str
    resolved_at: str
    ttd_minutes: float
    tta_minutes: float
    ttr_minutes: float
    action_items: int
    action_items_completed: int

# Monthly incident summary
monthly_incidents = [
    IncidentMetrics("SEV1", "2025-03-10 09:15", "2025-03-10 09:25", "2025-03-10 10:00",
                    ttd_minutes=7, tta_minutes=3, ttr_minutes=45, action_items=6, action_items_completed=5),
    IncidentMetrics("SEV2", "2025-03-18 14:30", "2025-03-18 14:33", "2025-03-18 15:00",
                    ttd_minutes=2, tta_minutes=3, ttr_minutes=30, action_items=3, action_items_completed=3),
]

avg_mttd = sum(i.ttd_minutes for i in monthly_incidents) / len(monthly_incidents)
avg_mttr = sum(i.ttr_minutes for i in monthly_incidents) / len(monthly_incidents)
action_completion = sum(i.action_items_completed for i in monthly_incidents) / sum(i.action_items for i in monthly_incidents)

print(f"Average MTTD: {avg_mttd:.1f} minutes")
print(f"Average MTTR: {avg_mttr:.1f} minutes")
print(f"Action item completion: {action_completion:.0%}")
```

---

## 7. Incident Response Tooling

### 7.1 Tool Stack

| Category | Tools | Purpose |
|----------|-------|---------|
| **Alerting** | PagerDuty, Opsgenie, Grafana OnCall | Route alerts to on-call engineers |
| **Communication** | Slack (incident channel), Zoom/Meet | Real-time collaboration |
| **Status page** | Statuspage.io, Cachet, Instatus | External customer communication |
| **Documentation** | Confluence, Notion, Google Docs | Postmortem writing and storage |
| **Tracking** | Jira, Linear, GitHub Issues | Action item tracking |
| **Automation** | Rundeck, PagerDuty Automation | Automated diagnostic and mitigation actions |

### 7.2 Incident Channel Bot

```python
"""Slack incident bot: automates incident channel creation and management."""

def create_incident_channel(severity: str, title: str, commander: str) -> str:
    """Create a Slack incident channel with standard setup."""
    channel_name = f"inc-{datetime.now():%Y%m%d}-{title.lower().replace(' ', '-')[:30]}"

    channel = slack.conversations_create(name=channel_name)

    # Post incident template
    slack.chat_postMessage(
        channel=channel["id"],
        text=f"""
:rotating_light: *Incident Declared: {title}*
*Severity*: {severity}
*IC*: <@{commander}>
*Status*: Investigating

*Quick Links:*
- <{grafana_url}|Grafana Dashboard>
- <{runbook_url}|Runbooks>
- <{statuspage_url}|Status Page Admin>

*Roles needed:*
:white_check_mark: IC: <@{commander}>
:question: Tech Lead: (volunteer or assign)
:question: Communication Lead: (volunteer or assign)
:question: Scribe: (volunteer or assign)

React with :eyes: to join the incident response.
        """
    )

    # Set channel topic
    slack.conversations_setTopic(
        channel=channel["id"],
        topic=f"{severity} | {title} | IC: @{commander} | Status: Investigating"
    )

    return channel_name
```

---

## 8. Building an Incident Response Culture

### 8.1 Cultural Practices

| Practice | Why It Matters |
|----------|---------------|
| **Celebrate good incident response** | Reinforces the behavior you want to see |
| **Share postmortems broadly** | Cross-team learning multiplies the value |
| **Track action item completion** | Postmortems without follow-through are waste |
| **Run game days** | Practice incident response before real incidents |
| **Reward learning, not blame** | People hide problems in blame cultures |
| **Measure and improve MTTR** | What gets measured gets improved |

### 8.2 Game Days

```
Game Day Plan: Payment Service Failure Simulation
─────────────────────────────────────────────────
Objective: Test the team's ability to detect, respond to, and mitigate
           a payment service dependency failure.

Setup (done by game master, not revealed to on-call):
  1. Inject 30-second latency into Stripe API calls (using toxiproxy)
  2. Start at 10:00 AM on a Tuesday (not Friday!)

Expected sequence:
  10:00 - Injection starts
  10:03 - Alert should fire (payment error rate > threshold)
  10:05 - On-call acknowledges, starts investigation
  10:10 - Root cause identified (Stripe latency)
  10:15 - Mitigation applied (circuit breaker, fallback processor)
  10:20 - Service recovered

Evaluation criteria:
  - Was MTTD < 5 minutes?
  - Were roles assigned promptly?
  - Was the runbook followed?
  - Was communication clear?
  - Were stakeholders notified?

Post-game review:
  - What surprised the team?
  - What runbook steps were missing or wrong?
  - What tools were missing or hard to use?
  - Update runbooks and alerts based on findings
```

---

## 9. Next Steps

- [27_AIOps_Anomaly_Detection.md](./27_AIOps_Anomaly_Detection.md) -- ML-based anomaly detection and intelligent alerting
- [28_Capstone_Full_Stack_Observability.md](./28_Capstone_Full_Stack_Observability.md) -- End-to-end observability platform design

---

## Exercises

### Exercise 1: Severity Classification

Classify each scenario with a severity level (SEV1-SEV4) and justify your decision:

1. A typo on the marketing homepage ("Recieve" instead of "Receive")
2. The login service returns 503 for 30% of requests
3. A database migration accidentally deletes a column, causing data loss for the last 2 hours of user registrations
4. The internal wiki is down during business hours
5. Credit card numbers are appearing in application logs (discovered during routine log review)

<details>
<summary>Show Answer</summary>

**1. Marketing homepage typo → SEV4**
- No functional impact
- Cosmetic issue with easy fix
- No user workflow affected
- Fix during normal business hours

**2. Login service 503 for 30% of requests → SEV2**
- Core functionality (authentication) significantly degraded
- 30% of users affected (between 10-50%)
- Workaround: users can retry (may succeed on a healthy instance)
- Page on-call, IC leads response

**3. Database migration data loss → SEV1**
- Data loss is ALWAYS SEV1 regardless of scope
- User registrations lost for 2 hours -- cannot be recovered from the application
- Revenue impact (lost registrations = lost customers)
- Requires immediate response: stop the bleeding, assess recovery options (backups)

**4. Internal wiki down → SEV3 (or SEV4)**
- No external user impact
- Internal productivity impact during business hours
- Workaround: use cached pages, ask colleagues
- Investigate during business hours, no page needed

**5. Credit card numbers in logs → SEV1**
- Security and compliance breach (PCI-DSS violation)
- Security incidents are ALWAYS SEV1 regardless of current user impact
- Requires immediate action: stop logging PII, purge existing logs, assess exposure
- May require regulatory notification depending on jurisdiction
- Even though discovered during routine review (not actively exploited), the exposure itself is critical

</details>

### Exercise 2: Postmortem Writing

An incident occurred with these facts:
- Service: search-service
- Duration: 2 hours (06:00 - 08:00 UTC, a Sunday)
- Impact: Search returned stale results (24 hours old) but did not error
- Root cause: The Elasticsearch reindexing cron job failed silently because the Elasticsearch cluster was in yellow status (one replica shard unassigned)
- Detection: A customer tweeted about outdated search results; the support team escalated
- No monitoring existed for index freshness

Write the postmortem action items section with at least 5 items, each categorized as prevention, detection, or mitigation.

<details>
<summary>Show Answer</summary>

| # | Action | Category | Owner | Priority | Due |
|---|--------|----------|-------|----------|-----|
| 1 | **Add search index freshness monitoring**: Create a Prometheus metric (`search_index_last_update_timestamp`) and alert if the index is more than 1 hour stale. | Detection | @search-team | P0 | 2025-03-18 |
| 2 | **Add Elasticsearch cluster health alerting**: Alert when cluster status is yellow or red for more than 10 minutes. Yellow status (missing replicas) degrades resilience and should be investigated. | Detection | @platform-team | P1 | 2025-03-20 |
| 3 | **Fix cron job error handling**: The reindexing cron job currently swallows errors silently. Add explicit error handling that: (a) logs structured error with Elasticsearch cluster status, (b) emits a `reindex_job_failure_total` counter metric, (c) sends a Slack notification to #search-alerts. | Prevention | @search-team | P0 | 2025-03-18 |
| 4 | **Add synthetic search freshness check**: Deploy a synthetic test that: (a) writes a known document to the source database, (b) waits 5 minutes, (c) searches for it in Elasticsearch, (d) alerts if not found. This catches freshness issues from any cause, not just cron failures. | Detection | @search-team | P1 | 2025-03-25 |
| 5 | **Fix the unassigned replica shard**: Investigate why the replica shard was unassigned (likely a node that left the cluster). Resize the Elasticsearch cluster or fix the node. Add a runbook for Elasticsearch shard allocation issues. | Prevention | @platform-team | P1 | 2025-03-20 |
| 6 | **Add user-facing staleness indicator**: When search results are older than the freshness SLO (1 hour), display a banner: "Search results may be outdated. We are working on refreshing them." This reduces customer-facing confusion while the team fixes the issue. | Mitigation | @frontend-team | P2 | 2025-04-01 |

**Key lessons:**
- Silent failures are the worst kind of failure -- the cron job should have screamed, not whispered.
- Freshness monitoring is as important as availability monitoring for data systems.
- Customer-reported issues (via Twitter) mean our detection failed completely -- MTTD was effectively 24 hours.

</details>

### Exercise 3: Runbook Design

Write a runbook for the alert "Database Connection Pool Exhausted" for a PostgreSQL-backed service. Include: overview, diagnostic steps with specific commands, mitigation actions, escalation path, and prevention measures.

<details>
<summary>Show Answer</summary>

```markdown
# Runbook: Database Connection Pool Exhausted

## Overview
**Alert**: DatabaseConnectionPoolExhausted
**Severity**: Critical (pages on-call)
**Service**: order-service
**Database**: PostgreSQL (orders-db)
**Dashboard**: [Database Health Dashboard](https://grafana.example.com/d/db-health)

This alert fires when the database connection pool utilization exceeds 90%
for more than 2 minutes. When the pool is exhausted, new requests queue and
eventually timeout, causing cascading 503 errors.

## Diagnostic Steps

### Step 1: Confirm the connection pool state
```bash
# Check current pool metrics
kubectl exec -it deploy/order-service -- curl localhost:8080/metrics | grep db_pool
# db_pool_active_connections 48
# db_pool_max_connections 50
# db_pool_waiting_requests 15
```

### Step 2: Check for long-running queries
```sql
-- Connect to PostgreSQL
kubectl exec -it statefulset/postgres-0 -- psql -U app -d orders

-- Find active queries running longer than 30 seconds
SELECT pid, now() - pg_stat_activity.query_start AS duration,
       query, state, wait_event_type, wait_event
FROM pg_stat_activity
WHERE state != 'idle'
  AND (now() - pg_stat_activity.query_start) > interval '30 seconds'
ORDER BY duration DESC;

-- Count connections by state
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
```

### Step 3: Check for connection leaks
```bash
# Check if connections are growing over time (leak indicator)
kubectl logs -l app=order-service --tail=200 | grep -i "connection\|pool\|leak"

# Check application error logs
kubectl logs -l app=order-service --tail=200 | grep ERROR
```

### Step 4: Check database server health
```sql
-- Check PostgreSQL max connections vs active
SELECT count(*) AS active, max_conn AS max
FROM pg_stat_activity,
     (SELECT setting::int AS max_conn FROM pg_settings WHERE name='max_connections') mc
GROUP BY max_conn;

-- Check for lock contention
SELECT blocked.pid AS blocked_pid,
       blocked.query AS blocked_query,
       blocking.pid AS blocking_pid,
       blocking.query AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_locks blocked_locks ON blocked.pid = blocked_locks.pid
JOIN pg_locks blocking_locks ON blocked_locks.locktype = blocking_locks.locktype
     AND blocked_locks.relation = blocking_locks.relation
     AND blocked_locks.pid != blocking_locks.pid
JOIN pg_stat_activity blocking ON blocking_locks.pid = blocking.pid
WHERE NOT blocked_locks.granted;
```

## Mitigation Actions

### If long-running queries found:
```sql
-- Kill the long-running query (use with caution)
SELECT pg_terminate_backend(<pid>);
```

### If application connection leak:
```bash
# Restart the application pods (rolling restart to release leaked connections)
kubectl rollout restart deploy/order-service
```

### If sudden traffic spike:
```bash
# Scale up application replicas (distributes connections across more pools)
kubectl scale deploy/order-service --replicas=10

# Temporarily increase pool size (if database can handle it)
kubectl set env deploy/order-service DB_POOL_MAX=100
```

### If database overloaded:
```bash
# Enable connection pooling proxy (PgBouncer)
kubectl scale deploy/pgbouncer --replicas=3
kubectl set env deploy/order-service DB_HOST=pgbouncer
```

## Escalation
- 10 minutes without resolution → page database team lead
- 20 minutes → page engineering director
- If data corruption suspected → page VP Engineering immediately

## Prevention
- Deploy PgBouncer as a connection pooler between applications and PostgreSQL
- Set connection pool max-lifetime to 5 minutes (prevents stale connections)
- Add connection leak detection in application health checks
- Set statement_timeout in PostgreSQL to 30 seconds (kill runaway queries)
- Monitor and alert on query duration percentiles
```

</details>

---

## References

- [PagerDuty Incident Response Guide](https://response.pagerduty.com/)
- [Google SRE Book -- Managing Incidents](https://sre.google/sre-book/managing-incidents/)
- [Etsy -- Blameless Postmortems](https://www.etsy.com/codeascraft/blameless-postmortems/)
- [Atlassian -- Incident Management Handbook](https://www.atlassian.com/incident-management/handbook)
- [PagerDuty -- On-Call Best Practices](https://www.pagerduty.com/resources/learn/on-call-best-practices/)
- [Jeli.io -- Incident Analysis](https://www.jeli.io/blog/category/incident-analysis/)
