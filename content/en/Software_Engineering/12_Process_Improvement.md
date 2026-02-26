# Lesson 12: Software Process Improvement

**Previous**: [Software Maintenance and Evolution](./11_Software_Maintenance_and_Evolution.md) | **Next**: [DevOps and CI/CD](./13_DevOps_CICD.md)

Every software organization has a process, whether it is defined or not. The question is not whether a process exists, but whether it is effective, visible, and improving over time. Software process improvement (SPI) is the disciplined practice of analyzing how software is built and operated, identifying weaknesses, and making targeted changes that lead to better outcomes — for quality, speed, cost, and team satisfaction.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Software Development Life Cycle (Lesson 02)
- Agile fundamentals (Lesson 03)
- Software quality assurance concepts (Lesson 07)
- Basic project management (Lesson 10)

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand why improving software processes produces better outcomes
2. Describe the CMM/CMMI maturity model and its five levels
3. Identify how ISO/IEC 12207 and 15504/33000 standards apply to process assessment
4. Apply PSP and TSP disciplines to personal and team-level improvement
5. Facilitate effective retrospectives using Start/Stop/Continue, 4Ls, and fishbone formats
6. Conduct root cause analysis using 5 Whys and fishbone diagrams
7. Design a metrics-driven improvement program using the GQM approach
8. Distinguish process tailoring from wholesale adoption of standard frameworks

---

## 1. Why Improve Software Processes?

### 1.1 The Cost of Poor Processes

Software failures are expensive. IBM research from the 1990s established that defects found in production are 100× more expensive to fix than defects found during requirements — a figure that has been refined but not overturned by subsequent studies. Process improvement addresses the systems that produce defects, not just individual defects themselves.

Poor processes manifest as:
- Requirements misunderstood and discovered late
- Builds that break on every merge
- Deployments that require a hero engineer who is the only person who knows the sequence
- Releases that slip repeatedly with no early warning
- Repetitive post-mortems for the same class of failure

### 1.2 The Case for Systematic Improvement

Individual heroism does not scale. A team where one exceptional engineer compensates for broken processes is fragile — when that person leaves, the team's performance collapses. Systematic process improvement creates:

- **Repeatability**: Results that do not depend on who is working that day
- **Predictability**: Reliable estimates because the process is understood
- **Learnability**: New team members can be productive faster
- **Visibility**: Problems surface early, when they are cheapest to fix

### 1.3 What Process Improvement Is Not

- It is not bureaucracy for its own sake. Processes that create work without reducing risk should be eliminated.
- It is not a one-time certification exercise. Improvement is continuous.
- It is not the same as adopting Agile (or SAFe, or DevOps). Frameworks are tools; improvement is a mindset.

---

## 2. CMM and CMMI

### 2.1 The Capability Maturity Model (CMM)

The Capability Maturity Model was developed at the Software Engineering Institute (SEI) at Carnegie Mellon University in the late 1980s, initially as a way for the U.S. Department of Defense to assess software contractors. It describes five maturity levels that organizations pass through as they improve their software processes.

| Level | Name | Characteristics |
|---|---|---|
| **1** | **Initial** | Chaotic. Success depends on individual heroism. No stable processes. |
| **2** | **Repeatable** | Basic project management established. Similar projects can be managed consistently. Requirements, schedule, and cost tracking in place. |
| **3** | **Defined** | Organization-wide standard processes. Tailoring guidelines for specific projects. Training programs. |
| **4** | **Managed** | Quantitative performance goals for both process and product quality. Statistical methods used to understand and control variation. |
| **5** | **Optimizing** | Continuous improvement through quantitative feedback. New ideas and technologies piloted in a disciplined way. |

Most commercial software organizations operate at Level 1 or 2. Achieving Level 3 is considered the threshold where process investment reliably pays off. Levels 4 and 5 require significant statistical process control infrastructure and are most common in defense, aerospace, and safety-critical domains.

### 2.2 CMMI: The Evolution

The **Capability Maturity Model Integration (CMMI)** superseded CMM in the early 2000s, integrating multiple CMMs (software, systems engineering, acquisition) into a single framework. CMMI v2.0 (released 2018) is the current version.

**CMMI has two representations:**

| Representation | Focus | Use Case |
|---|---|---|
| **Staged** | Organization-wide maturity level (1–5) | Benchmarking, contracts, DoD compliance |
| **Continuous** | Capability level (0–3) per process area | Targeted improvement, not certification |

**Key CMMI Process Areas (selected):**

| Process Area | Domain |
|---|---|
| Requirements Management | Managing changes to requirements |
| Project Planning | Establishing project plans |
| Project Monitoring and Control | Tracking performance against plan |
| Configuration Management | Controlling work products |
| Process and Product Quality Assurance | Compliance and quality audits |
| Causal Analysis and Resolution | Root cause identification and prevention |
| Organizational Process Focus | Planning and implementing process improvements |
| Quantitative Project Management | Statistical management of project performance |

### 2.3 Criticisms of CMMI

- Heavy documentation burden, especially at Levels 3–5
- Certification cost and effort can crowd out actual improvement work
- Waterfall-centric origins; tension with Agile practices
- "Maturity theater": organizations document processes that are not actually followed

CMMI is most valuable as a **lens for identifying gaps**, not as a rigid prescription.

---

## 3. ISO Standards for Software Processes

### 3.1 ISO/IEC 12207: Software Lifecycle Processes

ISO/IEC 12207 defines the processes, activities, and tasks involved in software development from conception through retirement. It establishes a common vocabulary for software lifecycle processes.

The standard organizes processes into three groups:

| Group | Process Examples |
|---|---|
| **Agreement Processes** | Acquisition, Supply |
| **Project Enabling** | Lifecycle Model Management, Infrastructure, Quality, Knowledge |
| **Technical Processes** | Requirements, Architecture, Design, Implementation, Integration, Verification, Validation, Operation, Maintenance |

ISO 12207 is framework-neutral — it describes **what** needs to happen, not **how**. An organization using Scrum can comply with ISO 12207 as long as its practices address the required process outcomes.

### 3.2 ISO/IEC 15504 / ISO 33000: Process Assessment

ISO/IEC 15504 (known as SPICE — Software Process Improvement and Capability dEtermination) provides a framework for assessing software process capability. It was superseded by the ISO 33000 series.

**Capability Levels in ISO 33000:**

| Level | Name | Characteristics |
|---|---|---|
| 0 | Incomplete | Process not performed or only partially |
| 1 | Performed | Process achieves its purpose |
| 2 | Managed | Process planned, monitored, and adjusted |
| 3 | Established | Process based on standard process with tailoring |
| 4 | Predictable | Process operates within defined limits using quantitative data |
| 5 | Optimizing | Process continuously improved to meet goals |

The key difference from CMMI: **ISO 33000 is process-area-specific**, not organization-wide. An organization can be at Level 4 for testing and Level 1 for requirements management, giving a more granular view than a single CMMI maturity level.

---

## 4. Conducting a Process Assessment

Before improving a process, you must understand its current state. A **process assessment** is a structured evaluation of an organization's software processes against a reference model (CMMI, ISO 33000, etc.).

### 4.1 Assessment vs. Appraisal

| Term | Scope | Purpose | Formality |
|---|---|---|---|
| **Assessment** | Internal, team-initiated | Identify improvement opportunities | Low (can be informal) |
| **Appraisal** | Formal evaluation | Obtain certification rating | High (audited, certified assessors) |

For most teams, internal assessments are far more useful than formal appraisals. The goal is learning, not certification.

### 4.2 The SCAMPI Method (CMMI Appraisals)

SCAMPI (Standard CMMI Appraisal Method for Process Improvement) is the official method for CMMI appraisals. Three classes exist:

| Class | Duration | Rigor | Typical Use |
|---|---|---|---|
| **SCAMPI A** | 1–3 weeks | Full; produces official rating | Certification for contracts |
| **SCAMPI B** | 3–5 days | Partial; identifies strengths/weaknesses | Pre-appraisal readiness check |
| **SCAMPI C** | 1–2 days | Lightweight; quick gap analysis | Internal improvement planning |

### 4.3 Lightweight Assessment Techniques

For teams that cannot justify a full SCAMPI appraisal, lightweight alternatives exist:

**Process questionnaire approach**: Team members independently complete a structured survey rating each process area on a 1–5 scale. Divergence between respondents reveals areas of unclear or inconsistently applied process.

```markdown
## Process Health Survey — Sample Items

Rate each on a scale of 1 (Not at all) to 5 (Consistently and well)

Requirements Management:
[ ] We have a defined process for capturing and recording requirements.
[ ] Requirements are reviewed and approved before development begins.
[ ] Changes to requirements go through a formal change control process.
[ ] Requirements are traceable to test cases.

Score: ___/20  Maturity Indicator: 1-8=Level 1, 9-14=Level 2, 15-20=Level 3

# Why compare scores between respondents?
# Divergence between team members reveals the real problem: not just process
# weakness, but inconsistent understanding of the process — a sign that the
# process is undocumented, unclear, or applied differently by different people.
```

**Walking the wall**: Have each team member explain their part of the process as if explaining it to a new hire. Gaps and contradictions in the narrative reveal process weaknesses.

**Value stream mapping session**: Map the actual end-to-end flow of a recent piece of work from idea to production, noting handoffs, wait times, and rework loops.

### 4.4 Assessment Findings Format

Well-documented assessment findings follow a consistent structure:

```markdown
## Process Area: Requirements Management
## Strength / Weakness: Weakness
## Evidence:
- In 4 of 5 sampled projects, requirements were not reviewed by the development team
  before the design sprint began.
- The project tracking system contains 23 open "requirements clarification" tasks
  older than 30 days.
## Impact:
- Late requirement clarification causes an average of 2-week rework cycles.
- Developer survey: 67% report "unclear requirements" as a top blocker.
## Recommended Action:
- Define a "requirements ready" checklist.
- Add a requirements review gate to the Definition of Ready for sprint planning.
- Owner: Product Manager | Target: End of Q2
```

---

## 5. Personal Software Process (PSP) and Team Software Process (TSP)

Developed by Watts Humphrey at the SEI, PSP and TSP bring process discipline to the individual and team levels.

### 4.1 Personal Software Process (PSP)

PSP teaches engineers to measure and manage their own work:

- **Size estimation**: Estimating lines of code or function points before starting
- **Effort estimation**: Time-in-phase estimates (planning, design, coding, review, testing)
- **Defect tracking**: Recording defect injection and removal by phase
- **Process scripts**: Structured checklists for design reviews, code reviews

**PSP data collection (simplified example):**

```markdown
## PSP Time Log — Feature: User Authentication

| Date     | Phase         | Start | End   | Delta | Interruptions |
|----------|---------------|-------|-------|-------|---------------|
| 2026-01-10 | Planning     | 09:00 | 09:30 | 30    | 0             |
| 2026-01-10 | Design       | 09:30 | 10:45 | 75    | 10            |
| 2026-01-10 | Code Review   | 10:45 | 11:15 | 30    | 0             |
| 2026-01-10 | Coding        | 11:15 | 13:00 | 105   | 15            |
| 2026-01-11 | Unit Test     | 09:00 | 10:00 | 60    | 0             |

## Defect Log

| ID | Date     | Injected Phase | Removed Phase | Defect Type | Fix Time |
|----|----------|----------------|---------------|-------------|----------|
| D1 | 2026-01-11 | Coding       | Unit Test     | Logic error | 20 min   |
| D2 | 2026-01-11 | Design       | Unit Test     | Interface   | 35 min   |
```

Over time, this data reveals personal patterns: which phases inject the most defects, how accurate estimates are, and where reviews catch the most issues.

**PSP's key insight**: Most software defects are injected by individual engineers, and most can be caught before testing through disciplined design and code review. The cost of finding a defect in review is dramatically lower than finding it in system test.

### 4.2 Team Software Process (TSP)

TSP extends PSP to the team level. TSP-managed projects:

- Begin with a launch phase where the team collectively defines goals, roles, and plans
- Use PSP data from individuals to produce team-level estimates
- Conduct weekly tracking meetings comparing planned vs. actual progress
- Report quantitative quality data (defect density, review rates) to management

TSP has demonstrated significant quality improvements in several studies, including a 2.7× reduction in defect density and schedule prediction accuracy within 5%. It requires significant organizational commitment to implement.

---

## 6. Retrospectives

Retrospectives are the most widely practiced process improvement technique in Agile organizations. Done well, they are where the team collectively examines its working process and makes concrete commitments to improve it.

### 5.1 Core Structure (Any Format)

A well-facilitated retrospective follows roughly this structure:

1. **Set the stage** (5 min): Establish safety, prime the team to reflect honestly
2. **Gather data** (15–20 min): What happened? Collect observations
3. **Generate insights** (10–15 min): Why did it happen? Find patterns
4. **Decide what to do** (10 min): What specific change will we make?
5. **Close** (5 min): Appreciation, next steps

### 5.2 Retrospective Formats

**Start / Stop / Continue**

The simplest and most widely used format. Team members write sticky notes for:
- **Start**: Things we should begin doing
- **Stop**: Things we should stop doing
- **Continue**: Things that are working and should be preserved

```
┌──────────────┬──────────────┬──────────────┐
│    START     │     STOP     │   CONTINUE   │
├──────────────┼──────────────┼──────────────┤
│ Daily design │ Skipping     │ Code reviews │
│ reviews      │ retrospects  │              │
│              │              │              │
│ Pair         │ Deploying on │ Automated    │
│ programming  │ Fridays      │ testing      │
│ for complex  │              │              │
│ features     │ Undocumented │ Team lunch   │
│              │ hotfixes     │ Wednesdays   │
└──────────────┴──────────────┴──────────────┘
```

**4Ls: Liked, Learned, Lacked, Longed For**

Useful when team morale is a concern or when you want a richer emotional dimension:
- **Liked**: What went well and felt good
- **Learned**: New insights gained this sprint
- **Lacked**: What was missing that would have helped
- **Longed For**: What you wish had happened

**Sailboat / Speedboat**

A visual metaphor: the team is sailing to an island (goal). Wind fills the sails (positive forces); anchors hold the boat back (negative forces); rocks are risks ahead.

### 5.3 Making Retrospectives Effective

- **Psychological safety**: People must feel safe raising concerns without fear of blame. The facilitator actively discourages blame.
- **Action items**: Each retrospective should produce 1–3 concrete, assigned, time-boxed action items. Retrospectives that produce lists of complaints with no follow-through lose credibility.
- **Follow up**: Begin each retrospective by reviewing the previous sprint's action items. Accountability closes the feedback loop.
- **Rotate the format**: Using the same format every sprint creates fatigue. Rotate formats quarterly.

---

## 7. Root Cause Analysis

When a significant failure occurs — a production outage, a major release delay, a critical security breach — root cause analysis (RCA) is used to understand the systemic cause, not just the surface event.

### 6.1 The 5 Whys

A deceptively simple technique: ask "why" repeatedly until you reach a root cause.

**Example: Production database outage**

```
Symptom: Database became unresponsive during peak traffic.

Why 1: Why did the database become unresponsive?
→ A long-running query locked the accounts table for 40 seconds.

Why 2: Why was a long-running query executed during peak hours?
→ A batch report was scheduled to run at 9 AM without considering business hours.

Why 3: Why was the report scheduled without considering business hours?
→ The developer who added the schedule did not know when peak traffic occurred.

Why 4: Why did the developer not know the traffic pattern?
→ There is no runbook or documentation for the production environment's peak periods.

Why 5: Why is there no such documentation?
→ There is no process requiring new developers to review operational context before
   deploying scheduled tasks.

Root Cause: No onboarding or process gate to ensure developers understand
             operational context before scheduling production workloads.

Corrective Actions:
1. Add peak-hour documentation to the developer onboarding checklist (Owner: Dev Lead, Due: 1 week)
2. Require ops review for any new scheduled task (Owner: Platform Team, Due: 2 weeks)
3. Add a query timeout advisory lock to prevent table-level locks > 5 seconds (Owner: DBA, Due: 3 weeks)
```

### 6.2 Fishbone (Ishikawa) Diagram

The fishbone diagram is a structured visual tool for brainstorming causes across multiple categories. The "head" of the fish is the effect (problem); the "bones" are categories of causes.

```
                                            EFFECT
   Causes                                 ┌────────────────┐
                                          │ Production DB  │
   People      Process                   │ Unresponsive   │
      \           \                       └────────────────┘
       \           \                              │
────────\────────────\────────────────────────────┘
         \            \
          \            \────── No review process for scheduled jobs
           \
            \────── Developer unaware of peak hours
                              ──────
   Technology    Environment
        \              \
         \              \────── No peak traffic documentation
          \
           \────── No query timeout configured
            \────── No table lock monitoring alert
```

**Standard fishbone categories for software:**

| Category | Examples |
|---|---|
| **People** | Skills, training, communication, staffing |
| **Process** | Missing steps, unclear procedures, handoff gaps |
| **Technology** | Tools, infrastructure, configuration, dependencies |
| **Environment** | Organizational pressures, culture, time constraints |
| **Data/Information** | Missing metrics, poor requirements, knowledge gaps |

### 6.3 Fault Tree Analysis

Fault Tree Analysis (FTA) works top-down, starting from the undesired event and systematically decomposing it into contributing causes using AND/OR logic gates. It is more formal than fishbone and is common in safety-critical systems.

```
              [Service Unavailable]
                       │
              OR Gate (any one cause)
               /       |        \
    [DB Failure]  [App Crash]  [Network Failure]
         │
    AND Gate (all required)
     /         \
[High Load]  [No Connection Pool]
```

FTA produces a quantitative probability model when combined with failure rate data.

---

## 8. Metrics-Driven Improvement: The GQM Approach

### 7.1 Goal-Question-Metric (GQM)

Developed by Victor Basili at the University of Maryland, GQM provides a top-down approach to defining metrics that are tied to actual improvement goals.

**Structure:**
1. **Goal**: A business or quality objective, stated in terms of purpose, object, quality attribute, viewpoint, and environment
2. **Questions**: Specific questions that must be answered to determine whether the goal is met
3. **Metrics**: Quantitative data that answers each question

**Example GQM model:**

```
# Why structure goals this formally?
# Without explicit Purpose/Object/Quality/Viewpoint, teams drift toward
# vanity metrics that look good on dashboards but don't drive improvement.
# The structured format forces clarity: "improve what, for whom, in what context?"

GOAL: Reduce the number of defects that escape to production
       [Purpose: Reduce] [Object: Defects] [Quality: Reliability]
       [Viewpoint: QA Team] [Context: Web platform, Q2 2026]

QUESTIONS:
  Q1: What is our current defect escape rate?
  Q2: At which phase are most escaping defects injected?
  Q3: Are code reviews detecting defects before testing?
  Q4: Which modules have the highest defect density?

METRICS:
  # Why tie every metric to a question?
  # Metrics without a clear question invite "measurement theater" — collecting
  # data nobody acts on. Each metric here directly answers a specific question
  # that determines whether the goal is being met.
  Q1 → M1: # production defects per release / # features released
  Q2 → M2: Defects by phase of injection (requirements/design/code)
  Q3 → M3: Defects found in code review / total defects found
  Q4 → M4: Defects per KLOC by module
```

### 7.2 The GQM Process

1. Define goals with stakeholders (not just engineers)
2. Derive 3–5 questions that operationalize each goal
3. Identify metrics that answer the questions — prefer objective, automated metrics
4. Collect baseline data before the improvement
5. Implement process changes targeted at improving the metrics
6. Measure again, compare, and decide next steps

**Anti-pattern to avoid**: Choosing metrics first, then working backward to justify them. This produces vanity metrics — numbers that look good but do not indicate actual improvement.

---

## 9. Continuous Improvement (Kaizen) in Software

**Kaizen** is the Japanese concept of continuous, incremental improvement. Applied to software, it means:

- No process is ever finished — always look for the next marginal improvement
- Small improvements are better than large infrequent ones
- Everyone on the team (not just management) is responsible for improvement
- Improvements are based on data, not opinions

### 8.1 The PDCA Cycle

Plan-Do-Check-Act (also called Deming Cycle) is the operational engine of continuous improvement:

```
    ┌─── Plan ───┐
    │            │
   Act          Do
    │            │
    └─── Check──┘

# Why a cycle, not a linear process?
# Improvement is never "done." Each Act feeds back into the next Plan.
# The cycle prevents both premature standardization (acting without checking)
# and analysis paralysis (planning without doing).

Plan:  Identify a problem and plan an improvement
Do:    Implement the improvement in a limited way (experiment)
Check: Measure the outcome against the expected result
Act:   If successful, standardize; if not, learn and plan again
```

**Software example:**
- **Plan**: Code review turnaround time is averaging 3 days. We believe this slows delivery. Hypothesis: if we create a "24-hour review SLA" norm, cycle time will improve.
- **Do**: Announce the SLA and track it for one sprint.
- **Check**: Average turnaround drops to 1.2 days. PR cycle time improves by 15%.
- **Act**: Add "24-hour review" to team norms document; track the metric monthly.

### 8.2 Value Stream Mapping

Value stream mapping (borrowed from lean manufacturing) visualizes the end-to-end flow of work from idea to production, making waste visible:

```
Idea → Backlog Refinement → Sprint Planning → Development → Code Review →
CI Pipeline → QA → Staging → Production

       [3 days]  [1 day]  [2 days]  [3 days]    [1 day]    [2 days]  [1 day]

# How to classify "value-added" vs "waste":
#   Value-added: Activities that directly transform the work product toward
#     what the customer wants (development, code review, CI pipeline).
#     Ask: "Would the customer pay for this step?"
#   Waste: Time where the work item is waiting, queued, or blocked — no
#     transformation is happening (backlog sitting, waiting for environment,
#     handoff delays, context switching between tasks).

Total lead time: ~13 days
Value-added time: ~6 days (development + review + pipeline)
  # Why these 3? They directly transform code: writing it, improving it, verifying it.
Waste: ~7 days (waiting, queue time, context switching)
  # Why is this waste? The work item exists but nobody is actively transforming it.
  # Reducing waste (e.g., eliminating approval queues) often yields bigger gains
  # than speeding up development itself.
```

Identifying and eliminating waste — waiting, rework, unnecessary approvals — is often more impactful than writing faster code.

---

## 10. Process Tailoring

No standard process framework fits every team perfectly. **Process tailoring** is the disciplined adaptation of a standard process to the specific constraints of a project.

### 9.1 Why Tailor?

- A 3-person startup does not need a formal change control board
- A safety-critical medical device project cannot skip formal verification
- A team of 5 senior engineers does not need the same mentoring overhead as a junior-heavy team

### 9.2 Tailoring Guidelines

A good tailoring approach follows three steps:

1. **Start from a defined standard**: Document the baseline process (even if it is "Scrum by the book")
2. **Identify tailoring decisions**: Which practices are mandatory, which are optional, which are context-dependent?
3. **Document the rationale**: Record why each tailoring decision was made, so it can be re-evaluated later

```markdown
## Tailoring Record: Project Phoenix (Mobile App)

### Base Process: Company Standard SDLC v2.3

| Practice              | Standard | Decision  | Rationale                              |
|-----------------------|----------|-----------|----------------------------------------|
| Requirements Review   | Required | Required  | Safety implications of health features |
| Architecture Review   | Required | Reduced   | MVP: 1 arch + PM sign-off (not board)  |
| Code Review           | Required | Required  | Standard coverage rules apply          |
| Load Testing          | Required | Deferred  | MVP has <1000 users; revisit at beta   |
| Security Pen Test     | Required | Required  | Health data; regulatory requirement    |
| Post-Release Review   | Optional | Required  | First mobile release; high learning value |
```

### 9.3 When Not to Tailor

Some practices should never be tailored away for expediency:
- Security review for systems handling personal or financial data
- Version control (there is no excuse to not use it)
- Test automation for regression paths that have caused production incidents

---

## 11. Benchmarking Against Industry Standards

Benchmarking compares your organization's performance against peers or industry data to identify improvement opportunities.

### 10.1 DORA Metrics

The DevOps Research and Assessment (DORA) metrics are the most empirically validated benchmarks for software delivery performance, based on thousands of survey respondents in the *Accelerate State of DevOps* reports.

| Metric | Elite | High | Medium | Low |
|---|---|---|---|---|
| **Deployment Frequency** | Multiple/day | 1/day–1/week | 1/month–1/week | < 1/month |
| **Lead Time for Changes** | < 1 hour | 1 day – 1 week | 1 week – 1 month | > 1 month |
| **Change Failure Rate** | 0–5% | 5–10% | 10–15% | > 15% |
| **MTTR (Time to Restore)** | < 1 hour | < 1 day | 1 day – 1 week | > 1 week |

Teams that achieve Elite performance on these metrics also report higher organizational performance (profitability, market share, productivity).

### 10.2 Using Benchmarks Correctly

- **Benchmark for direction, not destination**: The goal is continuous improvement, not reaching a specific tier
- **Internal benchmarks matter too**: Compare your team's performance to your own historical baseline
- **Beware Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure." Deployment frequency is only meaningful if each deployment delivers value.

---

## Summary

Software process improvement is the systematic discipline of making software development better — not through mandate, but through measurement, learning, and incremental change. Key concepts include:

- **CMM/CMMI maturity levels** (1–5): From chaotic Initial to continuously Optimizing. Most value is realized between Levels 2–3.
- **ISO/IEC 12207**: Standard vocabulary for software lifecycle processes (framework-neutral)
- **ISO 33000/SPICE**: Process capability assessment at the process-area level (Levels 0–5)
- **PSP/TSP**: Individual and team-level measurement disciplines that reduce defect injection and improve estimation accuracy
- **Retrospectives**: The most accessible SPI tool — structured, regular reflection with concrete action items
- **Root cause analysis**: 5 Whys and fishbone diagrams address systemic causes, not surface symptoms
- **GQM**: Tie every metric to a stated goal and question — measure what matters, not what is convenient
- **Kaizen / PDCA**: Continuous, small improvements beat infrequent large overhauls
- **Process tailoring**: Adapt standards to context; document and justify every deviation
- **DORA metrics**: Industry-validated benchmarks for software delivery performance

The ultimate goal of process improvement is not compliance with a framework but a culture where every team member is empowered to identify problems, experiment with solutions, and learn systematically.

---

## Practice Exercises

1. **CMMI Self-Assessment**: Consider a software team you know well (your current team, a previous job, or a case study). For each of the following CMMI process areas, assess the current capability level (1–3) and justify your assessment with specific observations: (a) Requirements Management, (b) Project Planning, (c) Configuration Management, (d) Causal Analysis and Resolution. What would the team need to do to advance each area by one level?

2. **GQM Design**: Your team wants to improve the reliability of its production systems. Define a complete GQM model: state one goal formally (Purpose/Object/Quality Attribute/Viewpoint), derive three questions, and for each question identify one or two concrete metrics that can be collected automatically. Identify what data sources would be needed.

3. **Retrospective Facilitation**: You are facilitating a retrospective for a sprint that went badly: the team missed its sprint goal, a hotfix caused a 2-hour outage, and two team members were in open conflict during a planning session. Design the retrospective: choose a format, write the agenda with timings, list 3 facilitation techniques you will use to maintain psychological safety, and describe what a successful outcome looks like.

4. **Root Cause Analysis**: A critical API endpoint began returning 503 errors at 2:17 AM, affecting 12% of users for 47 minutes before the on-call engineer restarted the service. The service had been deployed at 11 PM. Use the 5 Whys to analyze this incident through at least 4 levels. Then draw a fishbone diagram for the same incident. What corrective actions would address the root cause?

5. **Process Tailoring**: You are the engineering lead for a team of 6 building a personal finance tracking app for consumers. Your company's standard SDLC process was designed for enterprise projects with formal change control boards, weekly steering committee meetings, and mandatory architecture review board sign-off. Create a tailoring record that adapts this process for your context. Identify at least 5 practices to reduce, 2 to maintain without change, and 1 to enhance beyond the standard.

---

## Further Reading

- **Books**:
  - Humphrey, W. (1995). *A Discipline for Software Engineering*. Addison-Wesley. (PSP original)
  - Chrissis, M., Konrad, M. & Shrum, S. (2011). *CMMI for Development, 3rd Ed.* Addison-Wesley. (The definitive CMMI guide)
  - Forsgren, N., Humble, J. & Kim, G. (2018). *Accelerate: The Science of Lean Software and DevOps*. IT Revolution. (DORA research)
  - Derby, E. & Larsen, D. (2006). *Agile Retrospectives*. Pragmatic Bookshelf.

- **Papers and Online Resources**:
  - Basili, V. & Rombach, H. D. (1988). "The TAME Project." *IEEE Transactions on Software Engineering*. (GQM origin)
  - [DORA State of DevOps Reports](https://dora.dev/): Annual survey with industry benchmarks
  - [SEI CMMI Institute](https://cmmiinstitute.com/): Official CMMI resources and appraisal guidance
  - [retrospectivewiki.org](http://retrospectivewiki.org/): Community collection of retrospective formats

- **Standards**:
  - ISO/IEC 12207:2017 — Systems and software engineering: Software life cycle processes
  - ISO/IEC 33001:2015 — Concepts and terminology for process assessment (ISO 33000 series)
  - ISO/IEC 15939:2007 — Software measurement process (GQM-based)

---

**Previous**: [Software Maintenance and Evolution](./11_Software_Maintenance_and_Evolution.md) | **Next**: [DevOps and CI/CD](./13_DevOps_CICD.md)
