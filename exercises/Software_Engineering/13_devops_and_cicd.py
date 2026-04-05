"""
Exercises for Lesson 13: DevOps and CI/CD
Topic: Software_Engineering

Solutions to practice problems from the lesson.
Covers pipeline design, deployment strategies, DORA metrics, postmortems, blameless culture.
"""


# === Exercise 1: CI/CD Pipeline Design ===

def exercise_1():
    """Design a CI/CD pipeline for a web application with database."""

    print("EXERCISE 1: CI/CD Pipeline Design")
    print("=" * 65)

    stages = [
        {
            "stage": "1. Checkout & Setup",
            "duration": "~30s",
            "checks": "Clone repo, install dependencies from lockfile",
            "failure": "Network error, corrupted lockfile, dependency resolution conflict",
            "justification": "Must happen first — all subsequent stages need the code and deps.",
        },
        {
            "stage": "2. Static Analysis (Lint + Type Check + SAST)",
            "duration": "~1-2 min",
            "checks": (
                "ESLint/Prettier (style), mypy/TypeScript (types), Bandit/Semgrep (security)"
            ),
            "failure": "Lint violations, type errors, security findings (SQL injection, hardcoded secrets)",
            "justification": (
                "Cheapest checks run first. No need to run expensive tests if code has "
                "syntax errors or security vulnerabilities."
            ),
        },
        {
            "stage": "3. Unit Tests",
            "duration": "~2-5 min",
            "checks": "pytest/Jest unit test suite with coverage reporting",
            "failure": "Test assertion failure, coverage below threshold (e.g., 85%)",
            "justification": (
                "Fast feedback on business logic correctness. Unit tests are isolated "
                "and don't need a database."
            ),
        },
        {
            "stage": "4. Build (Docker Image)",
            "duration": "~2-3 min",
            "checks": "Multi-stage Docker build, image size check, layer caching",
            "failure": "Build error, image exceeds size limit",
            "justification": (
                "Build after unit tests pass — no point building a Docker image for broken code."
            ),
        },
        {
            "stage": "5. Integration Tests",
            "duration": "~5-10 min",
            "checks": (
                "Spin up test database (PostgreSQL in Docker), run migrations, "
                "execute API integration tests against real DB"
            ),
            "failure": "Migration failure, API contract violation, timeout",
            "justification": (
                "Tests actual database interactions, ORM behavior, and API contracts. "
                "More expensive than unit tests, so runs after them."
            ),
        },
        {
            "stage": "6. Security Scan (Container + Dependencies)",
            "duration": "~2-3 min",
            "checks": "Trivy/Grype container scan, dependency vulnerability scan (Snyk/Safety)",
            "failure": "Critical CVE in base image or dependency",
            "justification": (
                "Scans the built image (not just source). Catches vulnerabilities in "
                "OS packages and transitive dependencies."
            ),
        },
        {
            "stage": "7. Deploy to Staging",
            "duration": "~3-5 min",
            "checks": "Push image to registry, deploy to staging environment, run smoke tests",
            "failure": "Deployment timeout, health check failure, smoke test failure",
            "justification": (
                "Validates the deployment process and real infrastructure before production."
            ),
        },
        {
            "stage": "8. Deploy to Production (Canary)",
            "duration": "~10-15 min",
            "checks": (
                "Deploy to 5% of production traffic. Monitor error rate, latency, "
                "and business metrics for 10 minutes. Auto-rollback if thresholds exceeded."
            ),
            "failure": "Error rate > 1%, p95 latency > 500ms, business metric anomaly",
            "justification": (
                "Canary catches issues that staging misses (real traffic patterns, "
                "data-dependent bugs). Auto-rollback minimizes blast radius."
            ),
        },
        {
            "stage": "9. Full Production Rollout",
            "duration": "~5 min",
            "checks": "Promote canary to 100% of traffic. Final health check.",
            "failure": "Health check failure triggers rollback to previous version",
            "justification": "Only reached if canary was healthy. The 'green light' step.",
        },
    ]

    total_min = "~30-45 min"
    print(f"\n  TOTAL PIPELINE DURATION: {total_min}")
    print(f"  FAIL-FAST PRINCIPLE: Cheapest, fastest checks run first.\n")

    for s in stages:
        print(f"  {s['stage']} ({s['duration']})")
        print(f"    Checks: {s['checks']}")
        print(f"    Failure: {s['failure']}")
        print(f"    Why here: {s['justification']}")
        print()


# === Exercise 2: Deployment Strategy Selection ===

def exercise_2():
    """Select deployment strategy for a payment service with schema change."""

    print("EXERCISE 2: Deployment Strategy — Payment Service")
    print("=" * 65)

    print("""
  CONTEXT:
  - Payment processing service with 99.99% uptime SLO
  - New version includes a database schema change
  - 99.99% = max 4.38 minutes downtime per month

  RECOMMENDED STRATEGY: Blue-Green Deployment with Expand-Contract Migration

  WHY BLUE-GREEN (not canary, not rolling):
  1. Zero-downtime required (99.99% SLO). Blue-green provides instant
     switchover and instant rollback by swapping traffic.
  2. Payment services cannot tolerate partial failures — either the new
     version works or it doesn't. Canary's 5% traffic split is risky
     because 5% of payments could fail during testing.
  3. The database schema change complicates rolling deployments because
     old and new code would need to coexist with different schemas.

  DATABASE SCHEMA CHANGE — THE EXPAND-CONTRACT PATTERN:

  The schema change is the hardest part. You CANNOT simply ALTER TABLE
  while traffic is flowing. Use the three-phase expand-contract approach:

  Phase 1: EXPAND (backward-compatible)
    - Add new columns/tables alongside old ones
    - Deploy code that writes to BOTH old and new schema
    - Old version reads from old schema
    - New version reads from new schema, writes to both
    - Deploy this version first (before the blue-green switch)

  Phase 2: MIGRATE
    - Backfill new columns with data from old columns
    - Verify data consistency
    - Switch traffic to blue-green (new version now serves all traffic)
    - Monitor for 24-48 hours

  Phase 3: CONTRACT (cleanup)
    - Remove old columns/tables
    - Remove dual-write code
    - Deploy final clean version

  COMPLICATIONS OF SCHEMA CHANGE:
  1. Rollback is harder: If the new version fails after schema migration,
     rolling back requires data migration in reverse.
  2. Dual-write overhead: Writing to both schemas adds latency and complexity.
  3. Data consistency: Backfill must handle edge cases (null values,
     concurrent writes during migration).
  4. Testing: Must test with production-scale data, not just test fixtures.

  ROLLBACK PLAN:
  - Blue-green gives instant traffic rollback (switch back to blue).
  - Schema rollback: if Phase 2 data is corrupted, restore from pre-migration
    backup (taken before Phase 1).
  - Feature flag: gate the new schema behavior behind a flag. Disable flag
    = instant behavioral rollback without infrastructure changes.
""")


# === Exercise 3: DORA Assessment ===

def exercise_3():
    """DORA metrics assessment and improvement plan."""

    print("EXERCISE 3: DORA Metrics Assessment")
    print("=" * 65)

    metrics = {
        "Deployment Frequency": {
            "value": "Once per month",
            "category": "Low",
            "elite": "Multiple per day",
        },
        "Lead Time for Changes": {
            "value": "3 weeks",
            "category": "Low",
            "elite": "Less than 1 hour",
        },
        "Change Failure Rate": {
            "value": "30%",
            "category": "Low",
            "elite": "0-15%",
        },
        "Mean Time to Recovery": {
            "value": "2 days",
            "category": "Low",
            "elite": "Less than 1 hour",
        },
    }

    print("\n  DORA METRIC ASSESSMENT:")
    print(f"  {'Metric':<28} {'Current':<18} {'Category':<10} {'Elite Benchmark'}")
    print(f"  {'-' * 80}")
    for metric, info in metrics.items():
        print(f"  {metric:<28} {info['value']:<18} {info['category']:<10} {info['elite']}")

    print("""
  OVERALL CLASSIFICATION: LOW PERFORMER
  All four metrics are in the Low category. This team is significantly
  below industry median.

  METRIC TO TARGET FIRST: Change Failure Rate (30%)

  WHY THIS METRIC FIRST:
  1. A 30% failure rate means nearly 1 in 3 deployments causes an incident.
     This creates fear of deploying, which explains the low deployment
     frequency (monthly) and long lead times (3 weeks of "hardening").
  2. Fixing CFR breaks the vicious cycle: fewer failures -> more confidence
     -> more frequent deployments -> shorter lead times.
  3. The other metrics will naturally improve as a consequence.

  RECOMMENDED PRACTICES TO REDUCE CFR:

  1. Automated Testing Pipeline (immediate)
     - Add unit tests (target 80% coverage)
     - Add integration tests for critical paths
     - Block merges that fail tests (branch protection)
     Expected impact: Catch 50%+ of defects before deployment

  2. Code Review Standards (immediate)
     - Require 1 approval before merge
     - Use a checklist: correctness, security, performance
     Expected impact: Catch design-level issues that tests miss

  3. Canary Deployments (month 2)
     - Deploy to 5% of traffic first
     - Automated rollback on error rate spike
     Expected impact: Failures affect 5% of users, not 100%

  4. Trunk-Based Development with Feature Flags (month 3)
     - Smaller, more frequent merges reduce blast radius
     - Feature flags decouple deploy from release
     Expected impact: Each deployment is smaller = less risk per deploy

  EXPECTED OUTCOME AFTER 3 MONTHS:
  - CFR: 30% -> 15% (reduce by half)
  - Deployment frequency: monthly -> biweekly (confidence enables frequency)
  - Lead time: 3 weeks -> 1 week (faster pipeline, less hardening needed)
  - MTTR: 2 days -> 4 hours (canary + rollback reduces recovery time)
""")


# === Exercise 4: Postmortem Writing ===

def exercise_4():
    """Postmortem for a Black Friday database outage."""

    print("EXERCISE 4: Postmortem — Black Friday Database Outage")
    print("=" * 65)

    print("""
  POSTMORTEM: Database Connection Pool Exhaustion — Black Friday 2025
  ====================================================================

  Incident ID:    INC-2025-BF-001
  Date:           2025-11-28 (Black Friday)
  Duration:       45 minutes (14:32 - 15:17 UTC)
  Severity:       SEV-1 (Revenue-impacting)
  Author:         SRE Team
  Status:         Resolved

  SUMMARY:
  The e-commerce site was unavailable for 45 minutes during peak Black Friday
  traffic. The root cause was database connection pool exhaustion caused by a
  sudden 5x traffic spike that exceeded the configured connection pool limit.

  TIMELINE:
  14:15  Traffic begins ramping up (TV ad aired at 14:00)
  14:25  Database connection count reaches 80% of pool limit (alert fires)
  14:32  Connection pool exhausted (100/100 connections). New requests fail.
         Site returns 503 errors. Revenue loss begins.
  14:35  On-call SRE paged. Begins investigating.
  14:42  SRE identifies connection pool exhaustion in Grafana dashboard.
  14:45  SRE attempts to increase pool size via config change. Requires restart.
  14:50  SRE restarts application with pool size = 200. New instance starts.
  14:55  Old connections not releasing (long-running queries blocking pool).
  15:00  SRE kills long-running queries manually.
  15:05  Connection pool begins draining. Some requests succeed.
  15:17  All instances healthy. Site fully operational. Incident resolved.

  ROOT CAUSE ANALYSIS (5 Whys):

  Why 1: Why did the site return 503 errors?
    -> Database connection pool exhausted; no connections available.

  Why 2: Why was the connection pool exhausted?
    -> Traffic spiked 5x above normal, each request held a DB connection.

  Why 3: Why did each request hold a connection for so long?
    -> Several slow queries (product catalog joins) held connections for
       5-10 seconds instead of the expected 50ms.

  Why 4: Why were the queries slow?
    -> The product catalog table had 500K rows without an index on the
       'category' column used in the WHERE clause. Under load, these
       queries caused full table scans.

  Why 5 (ROOT CAUSE): Why was there no index?
    -> The index was present in development but was missed during the
       last database migration. The migration script was not tested
       under load. The CI pipeline does not verify database indexes.

  IMPACT:
  - Duration: 45 minutes of full outage
  - Users affected: ~120,000 (all users during the period)
  - Revenue loss: Estimated $340,000 (based on average revenue/minute)
  - Reputation: Social media complaints; trending on Twitter for 20 minutes

  ACTION ITEMS:

  | # | Action                                         | Owner     | Priority | Due     |
  |---|------------------------------------------------|-----------|----------|---------|
  | 1 | Add missing index on products.category          | DBA       | P0       | Today   |
  | 2 | Add index verification to CI migration tests    | Platform  | P1       | 1 week  |
  | 3 | Implement connection pool auto-scaling           | SRE       | P1       | 2 weeks |
  | 4 | Add load test stage to CI pipeline (Black        | QA/SRE    | P2       | 1 month |
  |   | Friday traffic simulation)                      |           |          |         |
  | 5 | Set up slow query alerting (queries > 1s)        | SRE       | P1       | 1 week  |
  | 6 | Create runbook for connection pool exhaustion     | SRE       | P2       | 2 weeks |

  WHAT WENT WELL:
  - On-call SRE responded within 3 minutes of page
  - Grafana dashboards clearly showed the connection pool saturation
  - Communication to stakeholders was timely (Slack channel, status page)

  WHAT WENT POORLY:
  - Increasing pool size required a restart (no hot-reconfiguration)
  - No pre-Black Friday load test was conducted
  - The missing index had been present for 3 weeks without detection
""")


# === Exercise 5: Blameless Culture ===

def exercise_5():
    """Response to a team lead advocating for blame-based postmortems."""

    print("EXERCISE 5: Blameless Culture — Response to Team Lead")
    print("=" * 65)

    print("""
  CONTEXT: A team lead says: "Our postmortems always identify the person who
  made the mistake. We need accountability."

  RESPONSE:

  I understand the instinct — accountability is essential for a healthy team.
  But there is an important distinction between accountability and blame,
  and conflating them actually REDUCES accountability.

  ACCOUNTABILITY vs BLAME:

  Blame asks: "WHO did this wrong?"
  Accountability asks: "WHY did the system allow this to happen?"

  Blame focuses on individuals: "Alice pushed a bad config."
  Accountability focuses on systems: "Our deployment process allowed an
  unreviewed config change to reach production."

  WHY BLAMELESS POSTMORTEMS PRODUCE BETTER OUTCOMES:

  1. INFORMATION FLOW:
     When people fear punishment, they hide information. If Alice knows
     she will be named in a postmortem and face consequences, she will:
     - Delay reporting the incident (hoping to fix it silently)
     - Omit details that make her look bad
     - Stop volunteering to work on risky features
     The NEXT incident takes longer to detect and resolve because people
     are afraid to speak up.

  2. ROOT CAUSES ARE SYSTEMIC:
     In complex systems, incidents are almost never caused by a single
     person's error. They are caused by systemic factors:
     - Why was the bad config deployable? (No validation)
     - Why was there no rollback mechanism? (Process gap)
     - Why was Alice working at 2 AM on a config change? (Overwork)
     Blaming Alice addresses ZERO of these systemic issues. The same
     class of incident will recur — just with a different person.

  3. PSYCHOLOGICAL SAFETY:
     Amy Edmondson's research (Harvard, 1999) shows that high-performing
     teams have high psychological safety — people feel safe to admit
     mistakes, ask questions, and raise concerns. Google's Project
     Aristotle confirmed this: psychological safety was the #1 predictor
     of team effectiveness.

     Blame-based postmortems destroy psychological safety. When people
     see a colleague named and punished, they learn: "Don't take risks,
     don't admit errors, don't deploy on Fridays."

  4. ACCOUNTABILITY STILL EXISTS:
     Blameless does NOT mean consequence-free. It means:
     - We hold the SYSTEM accountable: what process failed?
     - We assign ACTION ITEMS with owners: who will fix the process?
     - We track whether actions are completed
     - If someone is repeatedly negligent (pattern, not incident), that
       is a management conversation — not a postmortem topic.

  WHAT TO DO INSTEAD:

  1. In postmortems, use the phrase "the engineer" or "the on-call," never
     individual names. Focus on the sequence of events, not the actor.

  2. Ask "why" five times (5 Whys), driving toward systemic causes.

  3. End every postmortem with concrete action items that improve the
     SYSTEM so that the same mistake cannot happen again, regardless of
     who is working that day.

  4. Celebrate people who report incidents quickly and transparently.
     Make them heroes for their honesty, not villains for their mistakes.

  The goal is not to find someone to punish. The goal is to build a
  system where the mistake cannot happen again.
""")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: CI/CD Pipeline Design", exercise_1),
        ("Exercise 2: Deployment Strategy", exercise_2),
        ("Exercise 3: DORA Assessment", exercise_3),
        ("Exercise 4: Postmortem Writing", exercise_4),
        ("Exercise 5: Blameless Culture", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
