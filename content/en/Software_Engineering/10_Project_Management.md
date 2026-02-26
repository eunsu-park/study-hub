# Lesson 10: Software Project Management

**Previous**: [Process Models and Agile](./03_Agile.md) | **Next**: [Software Maintenance and Evolution](./11_Software_Maintenance_and_Evolution.md)

Software projects have a notoriously high failure rate. Studies by the Standish Group (CHAOS Report) consistently show that roughly one-third of software projects are canceled before completion, and another half are significantly over budget or behind schedule. Effective project management is the discipline that separates projects that deliver value from those that become cautionary tales.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Basic understanding of software development processes
- Familiarity with Agile and traditional SDLC models (Lessons 02–03)
- Some experience working on a software project

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the project management triangle and its implications for decision-making
2. Learn how to plan a software project: scope, schedule, and resources
3. Apply risk management techniques including risk registers and response strategies
4. Track project progress using Earned Value Management (EVM)
5. Distinguish between traditional (plan-driven) and Agile project management
6. Identify common causes of project failure and mitigation strategies

---

## 1. What Is Software Project Management?

Software project management is the process of planning, organizing, directing, and controlling resources to achieve specific software development goals within defined constraints. It bridges the technical work of engineering with the organizational reality of budgets, timelines, and human teams.

A **project** is a temporary endeavor with:
- A defined beginning and end
- A specific goal or deliverable
- Constraints on time, cost, and scope
- Uncertainty and risk

Project management is distinct from ongoing operations. Building a new e-commerce platform is a project; running it once deployed is operations. The skills overlap but are not identical.

### 1.1 Project Management Knowledge Areas (PMBOK)

The Project Management Body of Knowledge (PMBOK), published by the Project Management Institute (PMI), organizes project management into ten knowledge areas:

| Knowledge Area | Focus |
|---|---|
| Integration Management | Coordinating all project components |
| **Scope Management** | Defining what is (and isn't) in the project |
| **Schedule Management** | Planning and controlling the timeline |
| **Cost Management** | Estimating, budgeting, and controlling costs |
| Quality Management | Meeting quality standards |
| Resource Management | Acquiring and managing the team |
| Communications Management | Information flow among stakeholders |
| **Risk Management** | Identifying and responding to uncertainty |
| Procurement Management | Contracting with external suppliers |
| Stakeholder Management | Engaging those affected by the project |

Software projects interact with all ten areas, though scope, schedule, cost, and risk are often most critical.

---

## 2. The Project Management Triangle

The classic **iron triangle** (or triple constraint) states that every project is constrained by three factors:

```
           Scope
           /\
          /  \
         /    \
        /  Q   \
       /________\
     Cost       Time
```

- **Scope**: What the software must do (features, functionality, quality attributes)
- **Time**: The deadline or schedule for delivery
- **Cost**: Budget, including personnel, infrastructure, and licenses
- **Quality** is often added as a fourth dimension sitting inside the triangle

The fundamental truth of the triangle: **you can fix at most two of the three**. If a client demands fixed scope and a fixed deadline, cost must flex. If budget and schedule are fixed, scope must be negotiable. Ignoring this constraint is a root cause of many project failures.

### 2.1 Practical Implications

| Fixed | Fixed | Must Flex |
|---|---|---|
| Scope | Time | Cost (hire more people, use contractors) |
| Scope | Cost | Time (push the deadline) |
| Time | Cost | Scope (cut features, reduce quality) |

Brooks's Law (from *The Mythical Man-Month*) warns: "Adding manpower to a late software project makes it later." New team members need ramp-up time and increase coordination overhead. The triangle is real but not fully elastic.

---

## 3. Project Planning

Good planning does not mean predicting the future perfectly — it means creating a shared understanding of goals, constraints, and the current best path forward. Plans must be updated as new information arrives.

### 3.1 Scope Statement and Project Charter

The **project charter** is the founding document that authorizes the project and names the project manager. The **scope statement** defines:

- **Project objectives**: Measurable goals (e.g., "Reduce checkout time to under 2 seconds for 95th percentile users")
- **Deliverables**: Specific outputs (software release, documentation, trained users)
- **Acceptance criteria**: How will we know the deliverable is done?
- **Exclusions**: What is explicitly NOT in scope (critical to avoid scope creep)
- **Constraints**: Fixed deadlines, regulatory requirements, technology mandates
- **Assumptions**: Things believed to be true that have not been verified

```markdown
## Scope Statement Example: Customer Portal v2.0

### Objectives
- Replace legacy customer portal (ASP classic) with modern React/FastAPI stack
- Improve page load time by 50% vs current system
- Support 10,000 concurrent users (up from 2,000)

### Deliverables
- Production-ready web application
- API documentation (OpenAPI 3.0)
- Runbooks for operations team
- User acceptance test report

### Exclusions
- Mobile native app (future phase)
- CRM integration (separate project)
- Data migration of records older than 5 years

### Constraints
- Must be live before Q4 peak season (October 1)
- Budget: $450,000 USD
- Must comply with SOC 2 Type II requirements
```

### 3.2 Work Breakdown Structure (WBS)

A **Work Breakdown Structure** decomposes the project into manageable pieces. It is a hierarchical decomposition of deliverables, not activities.

```
Customer Portal v2.0
├── 1. Project Management
│   ├── 1.1 Project Planning
│   ├── 1.2 Status Reporting
│   └── 1.3 Project Closure
├── 2. Requirements
│   ├── 2.1 Stakeholder Interviews
│   ├── 2.2 Requirements Documentation
│   └── 2.3 Requirements Review
├── 3. Architecture and Design
│   ├── 3.1 System Architecture
│   ├── 3.2 API Design
│   └── 3.3 Database Schema
├── 4. Backend Development
│   ├── 4.1 Authentication Service
│   ├── 4.2 Customer API
│   ├── 4.3 Notification Service
│   └── 4.4 Background Jobs
├── 5. Frontend Development
│   ├── 5.1 Component Library
│   ├── 5.2 Customer Dashboard
│   ├── 5.3 Account Management
│   └── 5.4 Reporting Module
├── 6. Testing
│   ├── 6.1 Unit and Integration Tests
│   ├── 6.2 Performance Testing
│   └── 6.3 User Acceptance Testing
└── 7. Deployment
    ├── 7.1 Infrastructure Provisioning
    ├── 7.2 CI/CD Pipeline
    └── 7.3 Production Cutover
```

The lowest-level items in a WBS are called **work packages**. Each work package should be:
- Small enough to estimate accurately (typically 8–80 hours of effort)
- Assignable to a single owner
- Verifiable as complete

### 3.3 Scheduling

Once the WBS is defined, activities are sequenced with dependencies identified. Common techniques:

**Network Diagrams (Precedence Diagramming Method)**

```
[Requirements] → [Architecture] → [Backend Dev] → [Integration Test] → [UAT] → [Deploy]
                              ↘  [Frontend Dev] ↗
```

**Critical Path Method (CPM)**

The critical path is the longest sequence of dependent activities through the project. It determines the minimum project duration.

- **Early Start (ES)**: Earliest an activity can begin
- **Late Start (LS)**: Latest it can begin without delaying the project
- **Float/Slack**: LS - ES. Activities on the critical path have zero float.

Any delay on the critical path directly delays the project. Activities with float can slip without affecting the end date.

**Gantt Charts** provide a visual timeline view, mapping activities to calendar dates. They are effective for communication but can hide dependencies if not drawn carefully.

### 3.4 Resource Allocation

Resources include people (most important in software), infrastructure, tools, and budget. Key considerations:

- **Resource leveling**: Avoid over-allocating individuals. A developer assigned to three tasks simultaneously is not three times as productive.
- **Skills matching**: Assign work to people who have (or can develop) the necessary skills.
- **Velocity-based planning** (Agile): Use historical throughput to forecast how much work a team can complete per sprint.

---

## 4. Risk Management

Risk is the possibility of an uncertain event affecting project objectives. Unlike issues (which have already occurred), risks are future possibilities.

### 4.1 Risk Identification

Common techniques for surfacing risks:

- **Brainstorming**: Team workshops to enumerate potential risks
- **Checklists**: Standard risk categories for software projects
- **Expert interviews**: Domain specialists who have seen similar projects fail
- **Assumption analysis**: Every assumption is a potential risk if wrong
- **SWOT analysis**: Strengths, Weaknesses, Opportunities, Threats

Common risk categories in software projects:

| Category | Examples |
|---|---|
| **Technical** | Unfamiliar technology, integration complexity, performance unknowns |
| **Requirements** | Unclear or changing requirements, scope creep |
| **People** | Key staff turnover, skill gaps, team conflicts |
| **External** | Vendor delays, regulatory changes, third-party API changes |
| **Organizational** | Budget cuts, shifting priorities, insufficient stakeholder engagement |
| **Schedule** | Optimistic estimates, external dependencies, holidays/vacations |

### 4.2 Risk Analysis

Each identified risk is assessed on two dimensions:

- **Probability** (P): Likelihood the risk will materialize (typically 1–5 or Low/Medium/High)
- **Impact** (I): Consequence if it does (typically 1–5 or Low/Medium/High)

**Risk Score = Probability × Impact**

```
Impact
  5 | . . H H C
  4 | . M H H H
  3 | . M M H H
  2 | L L M M H
  1 | L L L M M
    +------------
      1 2 3 4 5  → Probability

L = Low, M = Medium, H = High, C = Critical
```

### 4.3 Risk Response Strategies

For each significant risk, a response strategy is chosen:

| Strategy | Description | When to Use |
|---|---|---|
| **Avoid** | Change the plan to eliminate the risk | When avoidance is feasible and not too costly |
| **Mitigate** | Reduce probability or impact | Most common strategy for technical risks |
| **Transfer** | Shift the risk to a third party | Insurance, fixed-price contracts, SLAs |
| **Accept** | Acknowledge the risk, plan contingency | When mitigation cost exceeds potential impact |

### 4.4 Risk Register

The risk register is the central artifact for tracking risks throughout the project.

```markdown
| ID  | Risk Description                    | P | I | Score | Strategy  | Response                          | Owner    | Status  |
|-----|-------------------------------------|---|---|-------|-----------|-----------------------------------|----------|---------|
| R01 | Key backend dev leaves mid-project  | 2 | 5 | 10    | Mitigate  | Cross-train 2nd dev on core APIs  | PM       | Active  |
| R02 | Third-party payment API deprecated  | 1 | 4 | 4     | Monitor   | Track vendor changelog monthly    | Tech Lead| Monitor |
| R03 | Requirements unstable (new VP)      | 3 | 4 | 12    | Avoid     | Lock requirements in contract     | PM       | Active  |
| R04 | Performance target unachievable     | 2 | 3 | 6     | Mitigate  | POC with load testing in week 2   | Architect| Active  |
| R05 | Deployment env not ready on time    | 3 | 4 | 12    | Transfer  | SLA with infrastructure team      | PM       | Active  |
```

Risk registers are living documents. Risks are added, updated, and closed throughout the project.

---

## 5. Stakeholder Management

**Stakeholders** are individuals or groups who affect, or are affected by, the project. Identifying and engaging stakeholders early is critical — a disengaged stakeholder discovered late can derail even technically successful projects.

### 5.1 Stakeholder Identification

Create a comprehensive stakeholder list:

- **Executive sponsor**: Funds the project, provides strategic direction
- **Product owner / business sponsor**: Defines requirements and priorities
- **End users**: Will use the delivered software daily
- **Development team**: Builds the software
- **Operations team**: Will run and maintain the system
- **Legal / compliance**: Ensures regulatory requirements are met
- **IT security**: Approves security architecture
- **External vendors**: Provide components or services
- **Customers** (if the software is commercial)

### 5.2 Stakeholder Analysis

The **Power/Interest grid** maps stakeholders to appropriate engagement strategies:

```
         High Power
              |
 Manage       |  Manage
 Closely      |  Closely
 (Key Players)|  (Satisfy)
              |
--------------+--------------  Interest
              |
 Monitor      |  Keep
 (Minimum     |  Informed
  Effort)     |  (Show/Tell)
              |
         Low Power
```

| Quadrant | Strategy |
|---|---|
| High Power / High Interest | Engage deeply, involve in decisions, frequent communication |
| High Power / Low Interest | Keep satisfied, don't overwhelm with detail |
| Low Power / High Interest | Keep informed, they are often advocates or detractors |
| Low Power / Low Interest | Monitor, communicate minimally |

### 5.3 Communication Plan

For each stakeholder group, define:
- **What** information they receive
- **How often** (daily standup, weekly status report, monthly steering committee)
- **In what format** (email, dashboard, meeting)
- **Who is responsible** for the communication

---

## 6. Progress Tracking: Earned Value Management

Earned Value Management (EVM) is a quantitative method for measuring project performance. It integrates scope, schedule, and cost into a unified picture.

### 6.1 Core EVM Metrics

| Metric | Symbol | Definition |
|---|---|---|
| **Planned Value** | PV | Budgeted cost of work scheduled (what we planned to spend by now) |
| **Earned Value** | EV | Budgeted cost of work performed (value of work actually done) |
| **Actual Cost** | AC | Actual cost incurred for work performed (what we actually spent) |
| **Budget at Completion** | BAC | Total project budget |

### 6.2 Performance Indices and Variances

| Metric | Formula | Interpretation |
|---|---|---|
| **Schedule Variance** | SV = EV − PV | Negative = behind schedule |
| **Cost Variance** | CV = EV − AC | Negative = over budget |
| **Schedule Performance Index** | SPI = EV / PV | < 1.0 means behind schedule |
| **Cost Performance Index** | CPI = EV / AC | < 1.0 means over budget |
| **Estimate at Completion** | EAC = BAC / CPI | Forecasted final cost |
| **Estimate to Complete** | ETC = EAC − AC | Remaining cost to finish |

### 6.3 EVM Example

A project has a budget (BAC) of $100,000 and is planned to be 50% complete at the end of month 3.

- **PV** = $50,000 (should have spent this much by now per plan)
- **EV** = $40,000 (only 40% of work is actually done)
- **AC** = $55,000 (we have actually spent this much)

Calculations:
```
SV = EV - PV = $40,000 - $50,000 = -$10,000  (behind schedule by $10k of value)
CV = EV - AC = $40,000 - $55,000 = -$15,000  (over budget by $15k)
SPI = EV / PV = 40,000 / 50,000 = 0.80  (80% of planned progress achieved)
CPI = EV / AC = 40,000 / 55,000 = 0.73  (earning 73 cents for every dollar spent)
EAC = BAC / CPI = 100,000 / 0.73 = $137,000  (project will cost ~$137k if trend continues)
```

This project is in trouble on both dimensions — an immediate review of root causes is warranted.

### 6.4 EVM in Agile Contexts

Traditional EVM requires detailed upfront planning. Agile EVM adaptations use:

- **Story points** as a proxy for value
- **Velocity** instead of resource-based estimates
- **Burndown/burnup charts** for sprint and release tracking
- **Cumulative flow diagrams** to identify bottlenecks

---

## 7. Team Management

### 7.1 Tuckman's Stages of Team Development

Bruce Tuckman's model describes how teams evolve:

| Stage | Characteristics | PM Response |
|---|---|---|
| **Forming** | Polite, uncertain, dependent on leader | Provide clear direction, structure, and goals |
| **Storming** | Conflict, power struggles, frustration | Facilitate resolution, establish norms, coach |
| **Norming** | Cohesion, shared methods, trust developing | Step back, encourage collaboration |
| **Performing** | High productivity, self-organizing, innovative | Delegate, remove obstacles, recognize achievements |
| **Adjourning** | Project ends, team disbands | Celebrate success, conduct retrospective, support transitions |

Teams do not progress linearly. Adding new members or changing conditions can push a team back to Forming or Storming.

### 7.2 Motivation Theories

**Maslow's Hierarchy of Needs** applied to software teams:

1. **Physiological**: Fair compensation, comfortable working conditions
2. **Safety**: Job security, predictable processes
3. **Social**: Team relationships, collaboration, inclusion
4. **Esteem**: Recognition, responsibility, professional growth
5. **Self-actualization**: Challenging work, autonomy, creative expression

**Herzberg's Two-Factor Theory** distinguishes:
- **Hygiene factors**: Salary, policies, working conditions — their absence causes dissatisfaction, but their presence does not motivate
- **Motivators**: Achievement, recognition, responsibility, growth — these drive engagement

Implication: Fixing hygiene problems removes demotivation. To genuinely motivate a team, focus on the work itself.

---

## 8. Agile Project Management vs. Traditional

Traditional (plan-driven) and Agile project management differ in fundamental assumptions:

| Dimension | Traditional (Waterfall/RUP) | Agile (Scrum/Kanban) |
|---|---|---|
| Planning horizon | Detailed upfront plan | Rolling wave, sprint-by-sprint |
| Change response | Formal change control process | Embrace change, reprioritize backlog |
| Progress measure | % tasks complete vs. plan | Working software (velocity, burndown) |
| Risk approach | Risk register, formal management | Short iterations reduce risk exposure |
| Documentation | Comprehensive, upfront | Lightweight, just-enough |
| Team structure | Functional, PM-directed | Cross-functional, self-organizing |
| Stakeholder engagement | Milestone reviews | Continuous, sprint reviews |

**Hybrid approaches** (SAFe, DSDM) combine elements of both for large organizations.

### 8.1 Scrum Roles and Artifacts

| Role | Responsibilities |
|---|---|
| **Product Owner** | Maintains and prioritizes product backlog |
| **Scrum Master** | Facilitates process, removes impediments |
| **Development Team** | Self-organizes to deliver sprint goals |

| Artifact | Purpose |
|---|---|
| **Product Backlog** | Ordered list of all desired work |
| **Sprint Backlog** | Work selected for current sprint |
| **Increment** | Working software produced each sprint |

---

## 9. Change Management and Scope Control

Scope creep — the gradual, uncontrolled expansion of project requirements — is one of the most common causes of schedule overruns and budget exhaustion. A formal **change management process** provides a gate through which all proposed scope changes must pass.

### 9.1 Change Control Process

```
                  ┌──────────────────┐
Change Request ──▶│ Change Log Entry │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Impact Analysis  │ ◀── Estimate: time, cost, risk, quality
                  └────────┬─────────┘
                           │
                     ┌─────┴──────┐
                     │  Decision  │
                     └─────┬──────┘
                      ╱         ╲
                 Approve       Reject/Defer
                    │               │
          ┌─────────▼─────┐  ┌──────▼────────┐
          │ Update: Plan, │  │ Notify         │
          │ Schedule,     │  │ Requestor with │
          │ Budget, WBS   │  │ Rationale      │
          └───────────────┘  └───────────────┘
```

### 9.2 Change Request Template

```markdown
## Change Request #CR-042

**Requested By**: Marketing Team   **Date**: 2026-03-15
**Priority**: Medium

### Description
Add a "share to social media" button on the order confirmation page.

### Business Justification
Expected 15% increase in social referrals based on A/B test data from competitor.

### Impact Analysis
| Dimension | Impact | Detail |
|-----------|--------|--------|
| Scope     | Low    | New component, isolated feature |
| Schedule  | +5 days | Design + dev + test + staging deploy |
| Cost      | +$3,200 | ~40 engineer hours at blended rate |
| Risk      | Low    | Third-party share SDK is well-documented |
| Quality   | None   | No regression risk to core checkout flow |

### Decision
- [ ] Approved (adjust baseline)
- [ ] Rejected
- [ ] Deferred to Phase 2
- [ ] Returned for more information

**Decision Made By**: Project Steering Committee   **Decision Date**: 2026-03-17
```

The key discipline: **no scope change without a formal decision that adjusts time, cost, or de-scopes something else**. "Just add it in" is how projects fail.

---

## 10. Project Closure

Project closure is an often-neglected phase. Teams frequently disband as soon as the software ships, leaving important administrative and organizational learning activities incomplete.

### 10.1 Closure Activities

| Activity | Purpose |
|---|---|
| **Formal acceptance** | Stakeholder sign-off confirming deliverables meet acceptance criteria |
| **Contract closeout** | Resolve open procurement items, release vendors |
| **Resource release** | Formally release team members back to their functional managers or next assignments |
| **Knowledge transfer** | Document institutional knowledge for the operations team |
| **Archive project artifacts** | Store project documents, code, and metrics in accessible form |
| **Lessons learned / post-mortem** | Capture what went well and what to improve for future projects |
| **Financial closeout** | Close purchase orders, finalize actuals, release uncommitted budget |

### 10.2 Lessons Learned Sessions

A **lessons learned session** (in Agile contexts, often a final retrospective) is a structured meeting to capture knowledge before the team disperses.

Best practices:
- Hold it within two weeks of project completion (memory fades quickly)
- Use a neutral facilitator, not the PM, to reduce defensiveness
- Focus on systems and processes, not people (blame-free)
- Capture both successes ("what to repeat") and failures ("what to avoid")
- Store the output where future PMs will actually find it

```markdown
## Lessons Learned: Customer Portal v2.0

### What Went Well
- Early load testing (week 2) caught a connection pool sizing issue before it became a crisis
- Daily 15-min standup kept the distributed team aligned across time zones
- User acceptance testing with real customers (not just internal stakeholders) caught 7 UX issues
  that would have required rework post-launch

### What to Improve
- API design should have been finalized before frontend development started;
  mid-project API changes caused 3 weeks of rework
- Risk R03 (unstable requirements from new VP) materialized but response was slow;
  need a faster escalation path for scope changes that break the baseline
- Documentation was written after code was complete rather than in parallel;
  this created a crunch in the final week

### Recommendations for Future Projects
1. Mandate API design freeze as a project milestone before frontend sprint begins
2. Add "stakeholder change velocity" as a leading risk indicator in risk reviews
3. Documentation sprints should run 1 week behind development, not at the end
```

---

## 12. Project Management Tools

| Tool | Type | Best For |
|---|---|---|
| **Jira** | Issue tracker + Agile boards | Software teams, Scrum/Kanban |
| **Linear** | Modern issue tracker | Fast-moving engineering teams |
| **Asana** | Task management | Cross-functional project tracking |
| **MS Project** | Traditional scheduling | Waterfall, EVM, Gantt charts |
| **Trello** | Kanban boards | Small teams, lightweight tracking |
| **GitHub Projects** | Code-integrated tracking | Open source, developer-centric |
| **Notion** | Flexible workspace | Documentation + lightweight PM |

Tool selection should follow team workflow, not the other way around.

---

## 13. Common Project Failures

### 10.1 The Standish Group CHAOS Findings

Based on thousands of IT projects:
- ~19% complete on time and on budget with full scope
- ~52% are challenged (late, over budget, or reduced scope)
- ~29% are canceled or fail outright

### 10.2 Root Causes of Failure

| Cause | Description |
|---|---|
| **Poor requirements** | Unclear, incomplete, or rapidly changing requirements |
| **Scope creep** | Uncontrolled addition of features without adjusting budget/schedule |
| **Unrealistic estimates** | Optimism bias, pressure from management to commit to impossible dates |
| **Poor communication** | Stakeholders not informed, team misaligned |
| **Key person dependency** | Single point of failure in critical knowledge |
| **Technical debt** | Shortcuts accumulate until velocity collapses |
| **Lack of executive support** | Project loses priority, resources are pulled |
| **Tool/technology mismatch** | Adopting unfamiliar technology without adequate ramp-up |

### 10.3 Lessons Learned

- **Under-promise, over-deliver**: Build buffers into estimates. Padding is not dishonesty — it is realism.
- **Define done**: Ambiguous acceptance criteria lead to "90% complete" projects that drag on indefinitely.
- **Kill projects early**: A project that should be canceled rarely recovers. Escalating commitment (sunk cost fallacy) keeps bad projects alive too long.
- **Celebrate learning from failure**: Post-mortems and retrospectives conducted without blame produce better outcomes than finger-pointing.

---

## Summary

Software project management balances scope, time, and cost while managing risk and stakeholders. Key practices include:

- **Project charter and scope statement**: Establish shared understanding of objectives and boundaries
- **WBS and scheduling**: Decompose work and identify the critical path
- **Risk management**: Identify risks early, assess probability × impact, and respond proactively with a risk register
- **Stakeholder management**: Map stakeholders on power/interest grid, tailor communication accordingly
- **Earned Value Management**: Track schedule (SPI) and cost (CPI) performance quantitatively
- **Team dynamics**: Understand Tuckman's stages; address hygiene factors first, then motivators
- **Agile vs. traditional**: Choose or blend approaches based on uncertainty and organizational context

Effective project management is not about following a process perfectly — it is about making informed decisions under uncertainty while keeping the team aligned and moving toward a shared goal.

---

## Practice Exercises

1. **Iron Triangle Trade-offs**: A client insists on full scope delivery in 6 months with a fixed budget of $200,000. Your estimate is 9 months at $280,000 for full scope. Write a one-page memo to the client that (a) explains the triple constraint, (b) presents three options with trade-offs, and (c) recommends one option with justification.

2. **Risk Register**: You are building a ride-sharing app with a 4-month deadline. Identify 8 risks across at least 4 categories (technical, people, external, requirements). For each, assign probability (1–3) and impact (1–5), calculate risk score, choose a response strategy, and describe a concrete response action.

3. **EVM Calculation**: A project has BAC = $200,000. After 4 months (of a planned 8), the project should be 50% complete (PV = $100,000). The team reports 42% of work is done and has spent $95,000 so far. Calculate: EV, SV, CV, SPI, CPI, and EAC. Interpret the results: is this project in good shape?

4. **WBS Design**: Create a WBS for building a personal finance tracking web application. The application allows users to log expenses, categorize spending, and view monthly summaries. Your WBS should have at least 3 levels and 20 work packages. For each work package, estimate the effort in hours.

5. **Stakeholder Analysis**: You are the PM for a hospital appointment scheduling system. Identify 10 stakeholders (internal and external). For each, determine their power (H/M/L), interest (H/M/L), and the engagement strategy you would use. Justify two cases where you would invest the most communication effort.

---

## Further Reading

- **Books**:
  - Brooks, F. (1995). *The Mythical Man-Month*. Addison-Wesley. (Timeless insights on software project management)
  - DeMarco, T. & Lister, T. (1999). *Peopleware: Productive Projects and Teams*. Dorset House. (The human side of software projects)
  - PMI. (2021). *A Guide to the Project Management Body of Knowledge (PMBOK Guide), 7th Ed*. (The standard reference)
  - Cohn, M. (2005). *Agile Estimating and Planning*. Prentice Hall.

- **Online Resources**:
  - [Standish Group CHAOS Reports](https://www.standishgroup.com/): Industry project success/failure statistics
  - [PMI Agile Practice Guide](https://www.pmi.org/pmbok-guide-standards/agile): Blending traditional PM with Agile
  - [EVM Tutorial](https://www.mitre.org/publications/systems-engineering-guide/): MITRE EVM reference

- **Tools to Explore**:
  - [Jira Software](https://www.atlassian.com/software/jira): Industry-standard Agile project tracking
  - [Linear](https://linear.app/): Modern engineering project management

---

**Previous**: [Process Models and Agile](./03_Agile.md) | **Next**: [Software Maintenance and Evolution](./11_Software_Maintenance_and_Evolution.md)
