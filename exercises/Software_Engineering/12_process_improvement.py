"""
Exercises for Lesson 12: Process Improvement
Topic: Software_Engineering

Solutions to practice problems from the lesson.
Covers CMMI assessment, GQM, retrospectives, root cause analysis, process tailoring.
"""


# === Exercise 1: CMMI Self-Assessment ===

def exercise_1():
    """CMMI self-assessment for a hypothetical team."""

    print("EXERCISE 1: CMMI Self-Assessment")
    print("=" * 65)

    print("""
  TEAM CONTEXT: Mid-size SaaS company, 15-person engineering team,
  building a project management tool. 2-week sprints, monthly releases.

  (a) REQUIREMENTS MANAGEMENT — Current Level: 2 (Managed)

  Observations:
  - Requirements are captured in Jira as user stories with acceptance criteria.
  - The Product Owner prioritizes the backlog regularly.
  - However, there is no formal traceability from requirements to tests.
  - Requirements changes are managed through backlog re-prioritization
    but not tracked with a formal change log.

  To Advance to Level 3 (Defined):
  - Implement a Requirements Traceability Matrix linking stories to test cases.
  - Define a standard requirements template with mandatory fields.
  - Establish a formal requirements review process before sprint planning.
  - Document the requirements management process in the team wiki.

  (b) PROJECT PLANNING — Current Level: 2 (Managed)

  Observations:
  - The team estimates using story points and plans sprints with velocity.
  - A project roadmap exists and is updated quarterly.
  - Risk identification happens informally during sprint planning.
  - No formal WBS or resource allocation model exists.

  To Advance to Level 3 (Defined):
  - Create a documented planning process (inputs, steps, outputs).
  - Maintain a formal risk register reviewed each sprint.
  - Calibrate story point estimates against historical actuals.
  - Establish planning metrics (velocity variance, forecast accuracy).

  (c) CONFIGURATION MANAGEMENT — Current Level: 3 (Defined)

  Observations:
  - All code is in Git with branch protection and required reviews.
  - CI/CD pipeline enforces build, test, and deployment automation.
  - Semantic versioning is used for releases.
  - Dependency lockfiles (package-lock.json) are committed.
  - However, infrastructure-as-code exists only partially.

  To Advance (Strengthen Level 3):
  - Complete IaC coverage (all infrastructure in Terraform).
  - Add SBOM generation to the CI pipeline.
  - Implement automated dependency vulnerability scanning.

  (d) CAUSAL ANALYSIS AND RESOLUTION — Current Level: 1 (Initial)

  Observations:
  - When incidents occur, the on-call engineer fixes the immediate issue.
  - Postmortems happen only for major outages (maybe 2-3 per year).
  - No systematic tracking of defect root causes or recurring issues.
  - Sprint retrospectives identify process issues but action items are
    rarely followed up.

  To Advance to Level 2 (Managed):
  - Run blameless postmortems for ALL incidents (not just major ones).
  - Track root causes in a database with categories (code, process, infra).
  - Review the root cause database monthly to identify patterns.
  - Assign owners and deadlines to retrospective action items; review
    completion at the next retrospective.
""")


# === Exercise 2: GQM Design ===

def exercise_2():
    """GQM model for production reliability improvement."""

    print("EXERCISE 2: GQM Model — Production Reliability")
    print("=" * 65)

    print("""
  GOAL (formal GQM format):
    Purpose:    Improve
    Object:     reliability of production systems
    Quality:    measured by availability and incident frequency
    Viewpoint:  from the perspective of the SRE team

  QUESTION 1: How often do production incidents occur?
    Metric 1a: Incident count per week (total, by severity)
      Data source: PagerDuty/Opsgenie incident log
    Metric 1b: Mean Time Between Failures (MTBF)
      Data source: Calculated from incident timestamps
      Formula: MTBF = Total uptime / Number of failures

  QUESTION 2: How quickly do we recover from incidents?
    Metric 2a: Mean Time to Recovery (MTTR) per incident
      Data source: PagerDuty (time from alert to resolution)
    Metric 2b: Percentage of incidents resolved within SLO (< 30 min)
      Data source: PagerDuty with SLO threshold filter

  QUESTION 3: Are we improving over time?
    Metric 3a: Incident count trend (4-week moving average)
      Data source: PagerDuty, charted in Grafana
    Metric 3b: Change failure rate (% of deployments causing incidents)
      Data source: CI/CD deployment log correlated with incident log
      Formula: CFR = Deployments causing incidents / Total deployments

  DATA COLLECTION PLAN:
    - PagerDuty API: Automated weekly export of incidents (severity, duration)
    - CI/CD system (GitHub Actions): Deployment timestamps and outcomes
    - Grafana dashboard: Auto-refreshed charts for all 5 metrics
    - Monthly review meeting: SRE team reviews trends and adjusts actions

  SUCCESS CRITERIA:
    After 3 months:
    - MTTR reduced by 25% (from current baseline)
    - Incident count (P1+P2) reduced by 30%
    - Change failure rate < 5%
""")


# === Exercise 3: Retrospective Facilitation ===

def exercise_3():
    """Design a retrospective for a difficult sprint."""

    print("EXERCISE 3: Retrospective for a Difficult Sprint")
    print("=" * 65)

    print("""
  CONTEXT:
  - Sprint goal missed
  - A hotfix caused a 2-hour outage
  - Two team members were in open conflict during planning

  FORMAT CHOSEN: Sailboat Retrospective
  (Visual metaphor: wind = things pushing us forward, anchor = things
  holding us back, rocks = risks ahead, island = our goal)

  Why this format: The sailboat metaphor is less confrontational than
  "what went wrong" formats. It encourages forward-looking thinking
  and treats obstacles as external forces, not personal failures.

  AGENDA (75 minutes):

  [0:00 - 0:05] OPENING: SET THE STAGE (5 min)
    - State the retrospective prime directive:
      "Regardless of what we discover, we understand and truly believe
      that everyone did the best job they could, given what they knew
      at the time, their skills and abilities, the resources available,
      and the situation at hand."
    - Briefly review sprint facts (goal, what was delivered, outage timeline)
    - Explicitly state: "This is a safe space. We focus on systems, not blame."

  [0:05 - 0:25] SILENT BRAINSTORMING: SAILBOAT CATEGORIES (20 min)
    - Each person writes sticky notes for 4 categories:
      Wind (what helped us): e.g., "Good pairing sessions"
      Anchor (what slowed us): e.g., "Unclear acceptance criteria"
      Rocks (risks ahead): e.g., "No test coverage for payment module"
      Island (our goal): e.g., "Ship v2.0 by end of Q1"
    - SILENT phase: no discussion, just writing. This ensures quieter
      voices contribute equally and prevents the conflict from dominating.

  [0:25 - 0:45] GROUP DISCUSSION: CLUSTER AND DISCUSS (20 min)
    - Facilitator reads notes aloud and groups by theme
    - For each cluster, ask: "Can someone elaborate on this?"
    - If the conflict topic arises, redirect to process:
      "What about our planning PROCESS contributed to the disagreement?"
      (Focus on process, not people)

  [0:45 - 0:65] DOT-VOTE AND ACTION ITEMS (20 min)
    - Each person gets 3 dots to vote on the most important anchors/rocks
    - Top 3 items become action items
    - For each action item: assign an owner and a "done by" date
    - Write actions on the team board (visible in the next sprint)

  [0:65 - 0:75] CLOSING (10 min)
    - Read back the 3 action items with owners
    - Quick round: "One word to describe how you feel about next sprint"
    - Thank the team for their honesty and participation

  FACILITATION TECHNIQUES FOR PSYCHOLOGICAL SAFETY:

  1. 1-2-4-All: For sensitive topics, first think individually (1), then
     pair-discuss (2), then group of 4, then share with all. This prevents
     the loudest voice from dominating and gives introverts space.

  2. Anonymous input: If the interpersonal conflict makes people hesitant,
     use anonymous digital sticky notes (Miro/FigJam) so people can raise
     issues without attribution.

  3. "Focus on the system" redirect: When discussion turns personal
     ("Alice should have..."), the facilitator says: "Let's focus on what
     our process could have done differently, not what individuals should
     have done." This is the most critical technique for this sprint.

  SUCCESSFUL OUTCOME:
  - 3 concrete, owned action items (not vague "be better" goals)
  - The team agrees on ONE process change to prevent the outage type
  - The conflicting team members either resolve the issue or agree to
    a follow-up 1-on-1 mediated by the Scrum Master
  - The team leaves feeling heard, not blamed
""")


# === Exercise 4: Root Cause Analysis ===

def exercise_4():
    """5 Whys and Fishbone diagram for an API outage."""

    print("EXERCISE 4: Root Cause Analysis — API Outage")
    print("=" * 65)

    print("""
  INCIDENT:
  Critical API endpoint returned 503 errors starting at 2:17 AM.
  12% of users affected for 47 minutes.
  On-call engineer restarted the service to restore it.
  Deployment at 11:00 PM the previous night.

  5 WHYS ANALYSIS:

  WHY 1: Why did the API return 503 errors?
    Because the application server ran out of memory and the OS killed
    the process (OOM killer).

  WHY 2: Why did the server run out of memory?
    Because a new code path (deployed at 11 PM) had a memory leak —
    it accumulated request objects in a list without clearing them.

  WHY 3: Why was the memory leak not caught before deployment?
    Because the team's test suite does not include memory/load tests.
    Unit tests pass even with memory leaks because they run briefly.

  WHY 4: Why does the team not have memory/load tests?
    Because the CI pipeline was set up for functional correctness only.
    Performance testing was considered "nice to have" and never prioritized.

  WHY 5 (ROOT CAUSE): Why was performance testing never prioritized?
    Because the team's Definition of Done does not include performance
    criteria. The deployment checklist only requires passing unit tests
    and code review — not load testing or memory profiling.

  FISHBONE DIAGRAM:

                                          [503 OUTAGE]
                                               |
    +-----------+----------+-----------+-------+-------+-----------+
    |           |          |           |               |           |
  [People]  [Process]  [Technology] [Environment]  [Methods]  [Measurement]
    |           |          |           |               |           |
    |           |          |           |               |           |
  No perf     No load    No memory   Late deploy    No canary   No memory
  testing     test in    monitoring  (11PM =        deployment  metrics in
  expertise   CI/CD      alerts      low staff)     strategy    dashboard
    |           |          |           |               |
  On-call     DoD does   OOM killer  No staging     Feature
  unfamiliar  not include logs not   env that       flag not
  with new    perf checks monitored  mimics prod    used for
  deployment             by alerting load           gradual
                                                    rollout

  CORRECTIVE ACTIONS (addressing root causes, not symptoms):

  1. IMMEDIATE: Add memory monitoring and OOM alerts to all services.
     Owner: SRE team. Due: This sprint.

  2. SHORT-TERM: Add a load/soak test to the CI/CD pipeline that runs
     for 10 minutes under simulated production load. Fail the build if
     memory growth exceeds 20%.
     Owner: Engineering lead. Due: Next sprint.

  3. MEDIUM-TERM: Update the Definition of Done to include:
     "Performance criteria verified (no memory leaks, latency within SLO)"
     Owner: Scrum Master. Due: Sprint planning meeting.

  4. LONG-TERM: Implement canary deployments. New code serves 5% of
     traffic for 30 minutes; automatic rollback if error rate or memory
     usage exceeds thresholds.
     Owner: Platform team. Due: End of quarter.
""")


# === Exercise 5: Process Tailoring ===

def exercise_5():
    """Tailor an enterprise SDLC for a small consumer app team."""

    print("EXERCISE 5: Process Tailoring — Consumer Finance App")
    print("=" * 65)

    print("""
  CONTEXT:
  - Team: 6 engineers building a personal finance tracking app
  - Audience: Consumers (not enterprise)
  - Company's standard SDLC: Designed for enterprise projects with:
    * Formal Change Control Board (CCB)
    * Weekly steering committee meetings
    * Mandatory Architecture Review Board (ARB) sign-off

  TAILORING RECORD:

  PRACTICES TO REDUCE (5):

  1. Change Control Board -> Product Owner Approval
     Standard: All changes require CCB review (3-5 people, weekly meeting).
     Tailored: Product Owner approves scope changes; technical changes
     approved by tech lead. CCB overhead is disproportionate for a 6-person team.
     Risk accepted: Less oversight; mitigated by CI/CD automated checks.

  2. Weekly Steering Committee -> Biweekly Sprint Review
     Standard: 1-hour weekly meeting with VP, PM, and architects.
     Tailored: Stakeholders attend the Sprint Review every 2 weeks.
     Asynchronous updates via a Slack channel for the off-weeks.
     Rationale: A consumer app has fewer stakeholders than enterprise.

  3. Architecture Review Board Sign-off -> Lightweight ADRs
     Standard: All architectural decisions require ARB approval (2-week lead time).
     Tailored: Team documents decisions as Architecture Decision Records (ADRs)
     reviewed by the tech lead. Major decisions (database choice, cloud provider)
     get a 30-minute architecture review with a senior architect.
     Rationale: Speed is critical for a consumer app. 2-week ARB delays kill velocity.

  4. Formal Requirements Specification (SRS) -> User Stories + Acceptance Criteria
     Standard: 30-page SRS document before development begins.
     Tailored: User stories with Gherkin acceptance criteria in Jira.
     Rationale: Requirements will change rapidly based on user feedback.
     An SRS would be outdated before it was finished.

  5. Formal Test Plan Document -> Automated Test Suite + Coverage Gate
     Standard: 20-page test plan signed by QA lead before testing begins.
     Tailored: pytest suite with 85% coverage gate in CI. No separate QA team;
     developers write tests. Exploratory testing before major releases.
     Rationale: A 6-person team does not have a dedicated QA function.
     Automated tests are the test plan.

  PRACTICES TO MAINTAIN WITHOUT CHANGE (2):

  1. Code Review (Mandatory PR Approval)
     Why: Code review catches bugs, spreads knowledge, and maintains code quality.
     This is valuable regardless of team size. Keep the requirement for
     1 approval before merge.

  2. Incident Response Procedure
     Why: Even a consumer app can have production incidents. The standard
     incident response (alert -> triage -> fix -> postmortem) is essential
     regardless of project type.

  PRACTICES TO ENHANCE BEYOND STANDARD (1):

  1. Deployment Frequency: ENHANCE from monthly (standard) to continuous
     Standard: Monthly release cycle with a 2-week hardening period.
     Enhanced: Continuous deployment — every merged PR is deployed to
     production via CI/CD with feature flags and canary rollouts.
     Why: A consumer app's competitive advantage is speed of iteration.
     User feedback loops must be days, not months. The standard monthly
     cycle would cripple the product's ability to respond to the market.
     Risk: Increased chance of production issues; mitigated by automated
     tests, feature flags, and automated rollback.
""")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: CMMI Self-Assessment", exercise_1),
        ("Exercise 2: GQM Design", exercise_2),
        ("Exercise 3: Retrospective Facilitation", exercise_3),
        ("Exercise 4: Root Cause Analysis", exercise_4),
        ("Exercise 5: Process Tailoring", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
