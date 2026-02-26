# Lesson 06: Estimation and Planning

**Previous**: [05. Software Modeling and UML](./05_Software_Modeling_and_UML.md) | **Next**: [07. Software Quality Assurance](./07_Software_Quality_Assurance.md)

---

Ask any software team how long the next project will take, and you will likely receive an optimistic answer that turns out to be wrong. Studies from the Standish Group's CHAOS reports consistently find that the majority of software projects run over schedule and over budget. This is not simply incompetence — estimation is genuinely hard because software is an intellectual product, teams are unique, and requirements change. This lesson equips you with the techniques, models, and habits of mind that produce better estimates and more realistic plans.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Lesson 02 — Software Development Life Cycles
- Lesson 03 — Agile and Iterative Development
- Lesson 04 — Requirements Engineering

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why software estimation is inherently uncertain and describe the Cone of Uncertainty
2. Apply Lines of Code (LOC) estimation and articulate its limitations
3. Describe Function Point Analysis (FPA) and compute a basic unadjusted function point count
4. Explain COCOMO and COCOMO II, and calculate a basic effort estimate using the COCOMO model
5. Use story points and relative estimation for agile sprint planning
6. Facilitate a Planning Poker session
7. Apply three-point estimation with the PERT formula
8. Construct a Work Breakdown Structure (WBS)
9. Build a Gantt chart and identify the critical path using the Critical Path Method (CPM)
10. Distinguish release planning from sprint planning
11. Recognize common estimation biases and apply techniques to counteract them

---

## Table of Contents

1. [Why Estimation Is Hard](#1-why-estimation-is-hard)
2. [Lines of Code and Its Limitations](#2-lines-of-code-and-its-limitations)
3. [Function Point Analysis](#3-function-point-analysis)
4. [COCOMO and COCOMO II](#4-cocomo-and-cocomo-ii)
5. [Story Points and Relative Estimation](#5-story-points-and-relative-estimation)
6. [Planning Poker](#6-planning-poker)
7. [T-Shirt Sizing](#7-t-shirt-sizing)
8. [Three-Point Estimation and PERT](#8-three-point-estimation-and-pert)
9. [Work Breakdown Structure](#9-work-breakdown-structure)
10. [Gantt Charts and Critical Path](#10-gantt-charts-and-critical-path)
11. [Release Planning vs. Sprint Planning](#11-release-planning-vs-sprint-planning)
12. [Estimation Accuracy: Tracking and Improving](#12-estimation-accuracy-tracking-and-improving)
13. [Common Estimation Pitfalls](#13-common-estimation-pitfalls)
14. [Summary](#14-summary)
15. [Practice Exercises](#15-practice-exercises)
16. [Further Reading](#16-further-reading)

---

## 1. Why Estimation Is Hard

### 1.1 The Cone of Uncertainty

The **Cone of Uncertainty** is a concept formalized by Barry Boehm and popularized by Steve McConnell. It describes how estimation accuracy improves as a project progresses.

```
  Estimated   │
  Duration    │  ╲                  4x overrun
  (ratio to  4│   ╲
   actual)    │    ╲
             3│     ╲
              │      ╲______________
             2│              ╲      ─ ─ ─ 2x overrun
              │               ╲
            1 │────────────────╲────────────── actual
              │                 ╲
           0.5│                  ╲_____________ 0.5x (ahead)
              │
              └──────┬────────────┬────────────┬────────
                Feasibility  Architecture  Feature    Done
                 Study        Complete     Complete
                 (±4x)        (±2x)        (±1.25x)
```

At project inception, estimates are accurate to within a **factor of four** in either direction. By the time architecture is complete, uncertainty narrows to ±2x. Only near feature completion does the error fall to ±25%.

Implications:
- Do not demand precise estimates before requirements are understood — they will be wrong
- Commit to estimates at the appropriate level of cone narrowing
- Re-estimate at each phase gate as more information becomes available

### 1.2 Fundamental Sources of Uncertainty

| Source | Description |
|---|---|
| **Requirements incompleteness** | Unknown scope leads to unknown work |
| **Technology novelty** | New frameworks, platforms, or languages have unknown characteristics |
| **Team unfamiliarity** | A team new to a domain or toolset is slower than historical data suggests |
| **Integration complexity** | Interfaces with external systems introduce hidden work |
| **Emergent design decisions** | Architecture choices mid-project change scope |
| **Human factors** | Illness, turnover, onboarding, meetings, and context-switching |

### 1.3 Estimation vs. Planning

A critical distinction:

- **Estimation** is a prediction of how long something will take given current knowledge
- **Planning** is a commitment about what will be delivered by when, accounting for business constraints

Estimation informs planning, but a plan can differ from an estimate when scope, resources, or schedule are adjusted. Never let a plan override an honest estimate without acknowledging the risk explicitly.

---

## 2. Lines of Code and Its Limitations

**Lines of Code (LOC)** — or KLOC (thousands of LOC) and MLOC (millions) — was the earliest and most intuitive size metric for software.

### 2.1 How LOC Is Used

Historically, organizations tracked productivity as LOC/person-month and defect density as defects/KLOC. Once a code base of similar type is known (from historical data), LOC can drive cost and schedule:

```
Effort = (LOC / Productivity Rate) × adjustment factors
```

Example: if the team's historical productivity is 1,000 LOC/person-month and the estimated size is 50,000 LOC:

```
Effort = 50,000 / 1,000 = 50 person-months
```

### 2.2 Limitations of LOC

| Problem | Explanation |
|---|---|
| **Language dependence** | 100 lines of Python may do the same as 400 lines of Java; LOC conflates verbosity with work |
| **Negative productivity** | Refactoring that removes 200 lines improves quality but counts as "negative productivity" |
| **Requires a design to estimate** | You must know what you are building before counting lines; circular for early estimation |
| **Incentivizes bloat** | Rewarding LOC encourages verbose, unmaintainable code |
| **Poor for modern development** | Configuration, infrastructure-as-code, and generated code distort counts |

LOC remains useful as a historical size metric for algorithmic, compute-heavy code (e.g., scientific computing, compiler internals) where lines map reasonably to complexity. For general business software, function points or story points are more appropriate.

---

## 3. Function Point Analysis

Function Point Analysis (FPA) was developed by Allan Albrecht at IBM in 1979. It measures **functional size** from the user's perspective — what the system does, not how it does it. This makes function points technology-independent.

### 3.1 The Five Function Types

Function points count five types of functional components:

| Function Type | Abbreviation | Description | Example |
|---|---|---|---|
| External Input | EI | Data entering the system from outside | Submit order form |
| External Output | EO | Data leaving the system to outside | Generate invoice |
| External Inquiry | EQ | Input/output pair with no persistent data change | Search products |
| Internal Logical File | ILF | Group of logically related data maintained by the system | Orders table |
| External Interface File | EIF | Group of data maintained by another system but referenced | Product catalog (from ERP) |

### 3.2 Complexity Classification

Each component is rated Low, Average, or High complexity, yielding a weight:

| Function Type | Low | Average | High |
|---|---|---|---|
| EI | 3 | 4 | 6 |
| EO | 4 | 5 | 7 |
| EQ | 3 | 4 | 6 |
| ILF | 7 | 10 | 15 |
| EIF | 5 | 7 | 10 |

### 3.3 Computing Unadjusted Function Points (UFP)

```
UFP = Σ (count × weight) for all function types
```

Example:

| Type | Count | Complexity | Weight | Subtotal |
|---|---|---|---|---|
| EI | 8 | Average | 4 | 32 |
| EO | 5 | Average | 5 | 25 |
| EQ | 6 | Low | 3 | 18 |
| ILF | 4 | Average | 10 | 40 |
| EIF | 2 | Low | 5 | 10 |
| **UFP** | | | | **125** |

### 3.4 Value Adjustment Factor

The **Value Adjustment Factor (VAF)** modifies UFP based on 14 General System Characteristics (GSC), each rated 0–5:

```
VAF = 0.65 + 0.01 × Σ(GSC scores)    [total GSC scores range 0–70]
VAF ranges from 0.65 to 1.35
```

```
Adjusted FP = UFP × VAF
```

ISO/IEC 20926 (IFPUG) and ISO/IEC 19761 (COSMIC) are standards-body variants of function point counting.

### 3.5 Converting FP to Effort

Once function points are counted, apply a productivity rate (FP/person-month) derived from historical data or industry benchmarks (e.g., ISBSG database):

```
Effort (person-months) = Adjusted FP / Productivity Rate
```

Industry benchmarks: 5–15 FP/person-month for business application development (varies widely by language, domain, and team).

---

## 4. COCOMO and COCOMO II

The **Constructive Cost Model (COCOMO)**, created by Barry Boehm in 1981, is an algorithmic model that estimates effort and schedule from the estimated size (in KLOC) of a software project.

### 4.1 COCOMO Basic Model

Three project modes (original 1981 model):

| Mode | Description | Example |
|---|---|---|
| **Organic** | Small team, well-understood problem, familiar environment | Business data processing |
| **Semi-detached** | Medium team, mixed experience, some novel requirements | Transaction processing system |
| **Embedded** | Tight hardware constraints, complex requirements, high reliability needed | Flight control software |

**Effort equation**:

```
E = a × (KLOC)^b    [person-months]
```

**Schedule equation**:

```
D = c × E^d    [months]
```

**Constants**:

| Mode | a | b | c | d |
|---|---|---|---|---|
| Organic | 2.4 | 1.05 | 2.5 | 0.38 |
| Semi-detached | 3.0 | 1.12 | 2.5 | 0.35 |
| Embedded | 3.6 | 1.20 | 2.5 | 0.32 |

**Example** (Organic, 32 KLOC):

```
E = 2.4 × (32)^1.05 = 2.4 × 36.5 ≈ 87.6 person-months
D = 2.5 × (87.6)^0.38 ≈ 2.5 × 14.0 ≈ 14.0 months
Average team size = E / D = 87.6 / 14.0 ≈ 6.3 people
```

### 4.2 COCOMO Intermediate

Intermediate COCOMO multiplies the basic effort estimate by **Cost Drivers** — 15 factors (grouped into product, computer, personnel, and project attributes), each rated Very Low to Extra High with a multiplier (Effort Multiplier, EM):

```
# Why multiply by the product of all EMs (Π)?
# Each cost driver independently scales effort. Multiplying captures their
# compounding effect — e.g., high reliability AND low analyst capability
# together amplify risk far more than either factor alone.
E = a × (KLOC)^b × Π(EM_i)
```

Example cost drivers:

| Driver | Very Low | Low | Nominal | High | Very High | Extra High |
|---|---|---|---|---|---|---|
| Required reliability (RELY) | 0.75 | 0.88 | 1.00 | 1.15 | 1.40 | — |
| Analyst capability (ACAP) | 1.46 | 1.19 | 1.00 | 0.86 | 0.71 | — |
| Use of software tools (TOOL) | 1.24 | 1.10 | 1.00 | 0.91 | 0.82 | — |

High reliability + low analyst capability + poor tooling can double or triple the basic estimate.

### 4.3 COCOMO II

COCOMO II (1995–2000) updated the model for modern development paradigms (object-oriented design, prototyping, COTS reuse, iterative development). Key improvements:

- **Size metric**: function points (converted to equivalent SLOC per language) or story points, not just LOC
- **Scale Factors**: five factors (Precedentedness, Development Flexibility, Architecture/Risk Resolution, Team Cohesion, Process Maturity) replace the mode categories
- **Three sub-models**:
  - *Application Composition* (early-phase, rapid prototyping)
  - *Early Design* (after architecture)
  - *Post-Architecture* (detailed design complete)

COCOMO II effort equation:

```
E = A × Size^SF × Π(EM_i)

where:
  A = 2.94 (calibrated constant)
  SF = B + 0.01 × Σ(scale factor scores)   [B = 0.91]
  Scale factor scores range 0–5; Σ ranges 0–25; SF ranges 0.91–1.23
```

COCOMO II is implemented in the open-source USC COCOMO II tool and commercially in Construx Estimate and SEER-SEM.

### 4.4 When to Use Algorithmic Models

Algorithmic models require:
1. Historical calibration data from similar projects
2. A size estimate before use
3. Careful selection of cost driver ratings

They are most appropriate for:
- Large, waterfall-style projects with stable requirements
- Government / defense contracts requiring documented cost justification
- Organizations with significant project history databases

For small teams or agile projects, story-point-based estimation is usually more practical.

---

## 5. Story Points and Relative Estimation

**Story points** are a relative, dimensionless unit used in agile development to express the effort required to implement a user story. Unlike hours or person-days, story points capture effort, complexity, and uncertainty together without committing to a specific duration.

### 5.1 Why Relative Estimation Works

Humans are poor at absolute estimation ("this will take 14 hours") but reasonably good at relative comparison ("story B is about twice as hard as story A"). Relative estimation exploits this cognitive strength.

### 5.2 Fibonacci Sequence for Story Points

Most teams use a modified Fibonacci sequence: **1, 2, 3, 5, 8, 13, 21, 40, 100**.

The increasing gaps between larger values reflect growing uncertainty: the difference between a 5 and an 8 is meaningful; the difference between a 40 and a 45 is not.

Some teams use: **XS, S, M, L, XL** (T-shirt sizes, see §7) or **powers of 2** (1, 2, 4, 8, 16).

### 5.3 Velocity

Once a team has completed several sprints, they observe their **velocity**: the average number of story points completed per sprint.

```
Velocity = Total story points completed in sprint / 1 sprint
           (averaged over several sprints for stability)
```

Velocity is used for **release planning**: given a backlog of N story points and a team velocity of V points/sprint, the project needs approximately N/V sprints.

### 5.4 Limitations of Story Points

- Points are team-relative: a "5" for Team A may be a "3" for a more experienced Team B
- Points measure complexity, not time; stakeholders who ask "how many hours is a story point?" misunderstand the model
- Velocity can be gamed by inflating estimates (a practice called "story point inflation")
- New teams have no historical velocity; initial sprints are used to calibrate

---

## 6. Planning Poker

Planning Poker is a consensus-based estimation technique that combines expert judgment, structured discussion, and the Delphi method to produce story point estimates.

### 6.1 The Process

1. **Preparation**: each team member receives a deck of cards with Fibonacci values (1, 2, 3, 5, 8, 13, 21, ?, ∞, ☕). The Product Owner reads a user story.

2. **Private selection**: each estimator privately selects a card representing their estimate. Crucially, cards are not revealed yet.

3. **Simultaneous reveal**: on a count of three, all estimators reveal their card at the same time. Simultaneous revelation prevents anchoring (the first number heard strongly biases all subsequent estimates).

4. **Discussion**: if estimates differ, the highest and lowest estimators explain their reasoning. This surfaces hidden complexity, misunderstood requirements, or different assumptions.

5. **Re-estimate**: the team discusses until convergence or votes again. This continues until estimates are within one Fibonacci step of each other.

6. **Record**: the agreed estimate is recorded against the story.

### 6.2 Special Cards

| Card | Meaning |
|---|---|
| `?` | "I don't understand the story well enough to estimate" → discussion needed |
| `∞` | "This story is too large to estimate; break it into smaller stories" |
| `☕` | "I need a break" |

### 6.3 Why Planning Poker Works

- **Structured debate**: forces articulation of assumptions
- **Anti-anchoring**: simultaneous reveal prevents cognitive bias
- **Team buy-in**: estimators who helped set the estimate are more committed to it
- **Knowledge sharing**: discussing a story surfaces implementation knowledge

### 6.4 Remote Planning Poker

Tools for distributed teams: PlanningPoker.com, Scrum Poker Online, Jira Planning Poker plugin, Miro templates.

---

## 7. T-Shirt Sizing

T-shirt sizing uses labels **XS, S, M, L, XL** (and sometimes XXL) to express relative size. It is faster than Planning Poker and is used:

- For **epics and themes** (too large for story points)
- In **early roadmap planning** when stories are not yet well defined
- As a **quick-filter** before committing to full Planning Poker

A mapping to story points is agreed upon by the team, for example:

| T-shirt | Story Points |
|---|---|
| XS | 1–2 |
| S | 3–5 |
| M | 8 |
| L | 13–21 |
| XL | 40+ (consider splitting) |

T-shirt sizing sacrifices precision for speed. It is a **first-pass estimate** to be refined as stories are elaborated.

---

## 8. Three-Point Estimation and PERT

Three-point estimation acknowledges that a task's duration is not a single number but a **distribution**. Three scenarios are estimated:

| Parameter | Symbol | Meaning |
|---|---|---|
| Optimistic | O | If everything goes well (best case; ~5th percentile) |
| Most Likely | M | The realistic expected duration |
| Pessimistic | P | If things go badly (worst case; ~95th percentile) |

### 8.1 PERT Formula

The **Program Evaluation and Review Technique (PERT)** uses a weighted average, giving most weight to the most-likely estimate:

```
Expected duration  E = (O + 4M + P) / 6
Standard deviation σ = (P - O) / 6
Variance         Var = σ²

# Why is M weighted 4×?
# PERT assumes a beta distribution for task duration. The most likely value (M)
# sits at the peak of the bell curve — it is the mode. Weighting it 4× reflects
# that outcomes cluster around M far more than at the tails (O or P).
# The divisor 6 normalizes across the full O + 4M + P = 6 "shares" of weight.
```

**Example**:

| Task | O | M | P | E | σ |
|---|---|---|---|---|---|
| Design schema | 2 | 3 | 8 | 3.33 | 1.0 |
| Implement API | 3 | 5 | 12 | 5.5 | 1.5 |
| Write tests | 1 | 2 | 5 | 2.33 | 0.67 |

For independent tasks in sequence, total expected duration and total variance are additive:

```
E_total = Σ E_i
Var_total = Σ Var_i
σ_total = √Var_total
```

For the above three tasks:
```
E_total = 3.33 + 5.5 + 2.33 = 11.17 days
σ_total = √(1.0² + 1.5² + 0.67²) = √(1 + 2.25 + 0.45) = √3.70 ≈ 1.92 days
```

A 90% confidence interval is approximately E ± 1.65σ = [11.17 ± 3.17] = [8.0, 14.3] days.

### 8.2 When to Use Three-Point Estimation

- For **individual task estimation** in detailed project plans
- For **risk quantification** when schedule uncertainty must be communicated to stakeholders
- Combined with **Monte Carlo simulation** for project-level uncertainty (run thousands of random samples from each task's distribution to generate a project completion probability distribution)

---

## 9. Work Breakdown Structure

A **Work Breakdown Structure (WBS)** is a hierarchical decomposition of the total project scope into manageable work packages. It answers the question: "What does the project need to produce?"

### 9.1 WBS Principles

- **Deliverable-oriented**: each node is a deliverable or outcome, not an activity
- **100% rule**: the WBS must account for 100% of the project scope — no more, no less
- **Mutually exclusive**: no work is counted twice
- **Work packages** (leaf nodes) are the smallest units, typically assigned to one person or team for one reporting period

### 9.2 WBS Example: Mobile Banking App

```
1. Mobile Banking App
├── 1.1 Project Management
│   ├── 1.1.1 Project Plans
│   ├── 1.1.2 Status Reports
│   └── 1.1.3 Risk Register
├── 1.2 Requirements
│   ├── 1.2.1 Stakeholder Interviews
│   ├── 1.2.2 Use Cases
│   └── 1.2.3 SRS Document
├── 1.3 Design
│   ├── 1.3.1 Architecture Document
│   ├── 1.3.2 Database Schema
│   └── 1.3.3 UI Wireframes
├── 1.4 Implementation
│   ├── 1.4.1 Authentication Module
│   ├── 1.4.2 Account Management Module
│   ├── 1.4.3 Transfer Module
│   └── 1.4.4 Notifications Module
├── 1.5 Testing
│   ├── 1.5.1 Unit Tests
│   ├── 1.5.2 Integration Tests
│   └── 1.5.3 UAT
└── 1.6 Deployment
    ├── 1.6.1 Infrastructure Setup
    └── 1.6.2 Production Release
```

The WBS is numbered using an **outline numbering** convention so each work package has a unique identifier (e.g., 1.4.3) that appears in schedules, budgets, and risk registers.

### 9.3 WBS Dictionary

Each work package should have a **WBS dictionary entry** containing:
- Description of work
- Responsible person / team
- Schedule (start, end)
- Estimated cost
- Dependencies
- Acceptance criteria

---

## 10. Gantt Charts and Critical Path

### 10.1 Gantt Charts

A **Gantt chart** is a bar chart that displays project tasks against a time axis. Each task is a horizontal bar; its length represents duration; its position represents the time window.

```
Task                   | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
-----------------------|-----|-----|-----|-----|-----|-----|
Requirements           |=====|=====|     |     |     |     |
Architecture           |     |  ===|=====|     |     |     |
Database Design        |     |     |=====|     |     |     |
Backend Development    |     |     |     |=====|=====|     |
Frontend Development   |     |     |  ===|=====|=====|     |
Testing                |     |     |     |     |=====|=====|
Deployment             |     |     |     |     |     |  ===|
```

Modern project management tools (Microsoft Project, Jira Plans, Asana, Linear) generate Gantt charts automatically from task dependencies and estimates.

### 10.2 Critical Path Method (CPM)

The **Critical Path** is the longest sequence of dependent tasks from project start to project end. Any delay on the critical path **directly delays the project**. Tasks not on the critical path have **float** (slack) — they can be delayed without impacting the project end date.

**Forward pass** — compute Earliest Start (ES) and Earliest Finish (EF):
```
# Why max? A task cannot start until ALL predecessors finish.
# The bottleneck predecessor (the one finishing last) dictates the earliest start.
ES(task) = max(EF of all predecessors)
EF(task) = ES(task) + duration
```

**Backward pass** — compute Latest Start (LS) and Latest Finish (LF):
```
# Why min? A task must finish before ALL successors need to start.
# The most demanding successor (earliest LS) dictates the latest allowable finish.
LF(task) = min(LS of all successors)
LS(task) = LF(task) - duration
```

**Float** (slack):
```
# Why does Float = 0 identify the critical path?
# Zero float means there is no scheduling flexibility — any delay propagates directly
# to the project end date. These tasks form the critical path.
Float = LS - ES = LF - EF
```

Tasks with Float = 0 are on the critical path.

**Example network**:

```
        ┌─────┐        ┌─────┐
  ●────►│ A:3 │───────►│ C:4 │──────►●
        └─────┘        └─────┘    (project end)
           │                          ▲
           │           ┌─────┐        │
           └──────────►│ B:6 │────────┘
                       └─────┘

Tasks: A (3 days), B (6 days), C (4 days, depends on A)
Path 1: A → C = 3 + 4 = 7 days
Path 2: A → B = 3 + 6 = 9 days  ← Critical Path
Float for C = 9 - 7 = 2 days
```

### 10.3 Crashing and Fast-Tracking

When the critical path is too long:
- **Crashing**: adding resources to critical-path tasks to shorten duration (costs money; diminishing returns due to Brooks's Law — see §13)
- **Fast-tracking**: executing critical-path tasks in parallel instead of sequentially (increases risk of rework)

---

## 11. Release Planning vs. Sprint Planning

### 11.1 Release Planning

**Release planning** determines which features will be delivered by a specific date (or which date a specific set of features will be ready). It operates at the **epic / story level** over multiple sprints.

Process:
1. Sort the backlog by priority (business value, risk, dependencies)
2. Determine team velocity (from historical sprints or initial calibration sprint)
3. Compute: Number of sprints = Total story points / Velocity
4. Assign stories to releases, drawing a **release boundary** at the point where scope meets capacity

```
 Sprint   | Stories         | Points | Cumulative | Release
----------|-----------------|--------|------------|--------
   1      | Login, Profile  |   18   |    18      |
   2      | Search, Filter  |   21   |    39      |  v1.0
   3      | Checkout, Cart  |   25   |    64      |
   4      | Payment, Review |   20   |    84      |  v1.1
   5      | Admin Panel     |   22   |   106      |
   6      | Reporting       |   18   |   124      |  v1.2
```

### 11.2 Sprint Planning

**Sprint planning** is a ceremony at the start of each sprint where the team selects stories from the backlog and commits to completing them within the sprint. It operates at the **story / task level** over days to a week.

Two parts:
1. **What?** — Product Owner presents top backlog items; team selects stories that fit within velocity
2. **How?** — Team breaks selected stories into engineering tasks (hours); identifies technical approach and dependencies

Outcome: **Sprint Backlog** — the set of stories and tasks the team will complete this sprint.

### 11.3 Iteration Zero (Sprint 0)

The first sprint in an agile project is often a "Sprint 0" for:
- Setting up development environment and CI/CD pipeline
- Establishing coding standards and branching strategy
- Doing initial architecture spike
- Running Planning Poker to calibrate velocity with a sample of backlog stories

This sprint does not deliver user-facing features but enables all subsequent sprints to deliver effectively.

---

## 12. Estimation Accuracy: Tracking and Improving

### 12.1 Tracking Actual vs. Estimated

For every sprint:

```python
# Pseudo-code for tracking estimation accuracy
# Why track accuracy per story? Because aggregate averages hide systematic
# mis-estimation of specific story sizes (e.g., 8-point stories consistently underestimated).
def sprint_report(sprint):
    accuracy_per_story = []
    for story in sprint.completed_stories:
        # For hour-based tasks
        accuracy = story.actual_hours / story.estimated_hours
        # Why this ratio direction (actual/estimated)?
        #   > 1.0 means over budget (underestimated the work)
        #   < 1.0 means under budget (overestimated the work)
        #   = 1.0 means perfect estimation
        # This convention makes it intuitive: values above 1 signal danger.
        accuracy_per_story.append(accuracy)

    avg_accuracy = mean(accuracy_per_story)
    # accuracy > 1.0: over-ran; < 1.0: finished early
    return avg_accuracy
```

Teams should track:
- **Velocity trend** (is it stable, improving, or declining?)
- **Accuracy ratio** (actual/estimated) per story size; often 8-point stories are consistently underestimated
- **Spillover rate** (percentage of committed stories not completed)

### 12.2 Calibration Techniques

| Technique | Description |
|---|---|
| **Historical analogy** | Compare new stories to previously completed stories of similar complexity |
| **Decomposition** | Break large stories into tasks; estimate tasks; sum |
| **Reference stories** | Maintain a "reference story" at each size point as a calibration anchor |
| **Wideband Delphi** | Structured expert consensus (the formal method Planning Poker approximates) |

### 12.3 Improving Estimation Over Time

1. **Hold retrospectives focused on estimation**: "Which stories did we mis-estimate most? Why?"
2. **Update reference stories**: as the team's skills grow, recalibrate what a "5" means
3. **Maintain an estimation checklist**: common forgotten tasks (code review time, documentation, deployment verification)
4. **Track estimation bias**: if the team consistently under-estimates by 20%, apply a 1.2 correction factor until behavior changes

---

## 13. Common Estimation Pitfalls

### 13.1 Optimism Bias (Planning Fallacy)

People systematically underestimate the time, costs, and risks of future actions. Daniel Kahneman termed this the **planning fallacy**: individuals predict completing tasks in the best-case scenario while ignoring historical rates and risks.

Countermeasure: **Reference class forecasting** — look at how long similar projects actually took before estimating the current one.

### 13.2 Anchoring Bias

The first number mentioned in an estimation discussion becomes a cognitive anchor. If a manager says "this should take about a week," all subsequent estimates are pulled toward that number.

Countermeasure: **Simultaneous reveal** (Planning Poker) and insisting estimates precede any discussion of desired timelines.

### 13.3 Parkinson's Law

*"Work expands to fill the time available for its completion."* If a task is given two weeks, it will take two weeks — even if it could have been done in three days.

Countermeasure: **Timeboxing** with explicit short durations and daily standups. Agile sprints naturally apply timeboxing.

### 13.4 Brooks's Law

*"Adding manpower to a late software project makes it later."* (Fred Brooks, *The Mythical Man-Month*) New team members require onboarding, increase communication overhead, and partition work in ways that create integration problems.

Countermeasure: Plan team composition upfront; avoid last-minute additions. If adding people is unavoidable, add them to non-critical work.

### 13.5 Student Syndrome

People delay starting tasks until the last moment, then rush — causing quality problems and spillover.

Countermeasure: Daily standups make progress visible; Definition of Done criteria prevent premature closure.

### 13.6 Ninety-Percent Syndrome

A task reported as "90% complete" tends to stay at "90% complete" for an extended time. The last 10% contains the hardest, most uncertain work.

Countermeasure: Use binary completion tracking (done / not done) for work packages. Report **remaining work**, not percentage complete.

### 13.7 Scope Creep

Uncontrolled growth in project scope without corresponding adjustment to schedule, budget, or resources.

Countermeasure: Formal change control (see Lesson 04); MoSCoW prioritization; sprint-level scope commitments.

---

## 14. Summary

Estimation is a skill that improves with deliberate practice, calibration data, and structured techniques.

| Technique | Best for | Key Formula |
|---|---|---|
| LOC | Algorithmic, compute-heavy code | `Effort = LOC / productivity` |
| Function Points | Technology-independent sizing | `UFP = Σ(count × weight)` |
| COCOMO | Large, waterfall projects | `E = a × KLOC^b × Π(EM)` |
| Story Points | Agile sprints | Relative; calibrated via velocity |
| Planning Poker | Team consensus estimation | Simultaneous reveal + discussion |
| T-shirt sizing | Epics, early roadmaps | XS/S/M/L/XL |
| Three-point / PERT | Task-level uncertainty | `E = (O + 4M + P) / 6` |
| WBS | Scope decomposition | 100% rule; outline numbering |
| Critical Path | Schedule optimization | Float = LS − ES |

Key principles:
- Acknowledge and communicate uncertainty; use the Cone of Uncertainty to set expectations
- Prefer **relative estimation** (story points) for agile teams; it is faster and more accurate than absolute time estimates
- **Track actuals** against estimates religiously; the data is the foundation of improved future estimates
- Protect estimates from anchoring, optimism bias, and management pressure

---

## 15. Practice Exercises

**Exercise 1: PERT Estimation**

A software team has identified three tasks for a new feature:

| Task | Optimistic | Most Likely | Pessimistic |
|---|---|---|---|
| Database migration | 1 day | 2 days | 6 days |
| API implementation | 3 days | 5 days | 10 days |
| Frontend integration | 2 days | 4 days | 9 days |

Assume the tasks are sequential (each depends on the previous).

a. Calculate the PERT expected duration and standard deviation for each task.
b. Calculate the total expected project duration and standard deviation.
c. Compute a 90% confidence interval for the project completion date.
d. The project manager says "I'll commit to 13 days." What probability of success does this represent? (Hint: compute the Z-score.)

---

**Exercise 2: Function Point Counting**

An online poll application has the following components:

- Create poll form (input: question + up to 10 options) — Average EI
- Cast vote form (input: poll ID + option choice) — Low EI
- View results page (output: bar chart of votes per option) — Average EO
- Search polls by keyword (input/output pair, no data change) — Low EQ
- Polls table (maintained by the system) — Average ILF
- Users table (maintained by the system) — Low ILF
- Authentication service (external system, referenced but not maintained) — Low EIF

Calculate the Unadjusted Function Points (UFP). If the team's historical productivity is 8 FP/person-month, estimate the effort in person-months.

---

**Exercise 3: Critical Path**

Given the following project network (task: duration, predecessors):

| Task | Duration | Predecessors |
|---|---|---|
| A | 4 days | — |
| B | 6 days | — |
| C | 3 days | A |
| D | 5 days | A, B |
| E | 4 days | C, D |
| F | 2 days | D |
| G | 3 days | E, F |

a. Draw the network diagram.
b. Perform forward and backward passes to compute ES, EF, LS, LF, and Float for each task.
c. Identify the critical path and the project duration.
d. If task D is delayed by 2 days, what is the new project duration?

---

**Exercise 4: Release Planning**

A product backlog contains 240 story points. The team's measured velocity over the past four sprints is: 28, 32, 30, 26 points per sprint (2-week sprints).

a. Calculate the team's average velocity.
b. Estimate the number of sprints and calendar months to complete the backlog.
c. The product owner wants to release the first 100 story points (sorted by priority) as v1.0. How many sprints will this take?
d. A senior engineer will be on leave for one sprint, reducing velocity by 20% for that sprint. How does this affect the v1.0 release date?

---

**Exercise 5: Estimation Pitfalls**

For each scenario below, identify the estimation pitfall(s) at work and suggest a mitigation:

a. The project manager announces a deadline of 6 months in the kickoff meeting before the team has done any estimation. All subsequent estimates cluster around 6 months.
b. The team reports a difficult database migration task as "95% complete" for three consecutive weeks.
c. A developer working alone on a large feature waits until day 9 of a 10-day sprint to start coding.
d. To meet a slipping deadline, the manager assigns three new developers to the project with two weeks remaining.
e. The team estimates features based on the best-case scenario ("if everything goes smoothly"), ignoring past sprints where integration testing repeatedly uncovered blocking bugs.

---

## 16. Further Reading

- McConnell, S. — *Software Estimation: Demystifying the Black Art* (Microsoft Press, 2006) — the most practical and readable book on the subject
- Boehm, B. — *Software Engineering Economics* (Prentice-Hall, 1981) — original COCOMO; foundational text
- Boehm, B. et al. — *Software Cost Estimation with COCOMO II* (Prentice-Hall, 2000) — COCOMO II in full
- Brooks, F. — *The Mythical Man-Month* (Addison-Wesley, 1975; anniversary edition 1995) — essential reading; Brooks's Law, the surgical team model, conceptual integrity
- Kahneman, D. — *Thinking, Fast and Slow* (Farrar, Straus and Giroux, 2011) — cognitive biases including planning fallacy and anchoring
- Cohn, M. — *Agile Estimating and Planning* (Prentice-Hall, 2005) — story points, Planning Poker, velocity-driven release planning
- IFPUG — *IFPUG Function Point Counting Practices Manual* (Release 4.3.1) — authoritative FPA reference
- PMI — *A Guide to the Project Management Body of Knowledge (PMBOK Guide)* — WBS, CPM, and earned value management

---

## Exercises

### Exercise 1: Apply the Cone of Uncertainty

A project manager is asked to commit to a launch date at the very beginning of a new project. Requirements have been captured in rough user stories but no architecture work has been done.

(a) According to the Cone of Uncertainty, what is the accuracy range of any estimate made at this point?
(b) What artifacts or milestones must be completed before the estimate narrows to ±2×?
(c) A stakeholder insists on a firm date. Write a one-paragraph response explaining why committing at this stage is risky and what you can offer instead.

### Exercise 2: Compute COCOMO Estimates

A new payroll system is classified as an Organic project with an estimated size of 24 KLOC. Use the COCOMO basic model formulas from Section 4.

(a) Calculate the expected effort in person-months.
(b) Calculate the expected schedule duration in months.
(c) Estimate the average team size.
(d) The requirements change mid-project and scope grows to 40 KLOC. Recalculate all three values. By what percentage did effort increase relative to the size increase?

### Exercise 3: Plan a Sprint Using Story Points

A team has a measured velocity of 34 points per two-week sprint. Their backlog for the next release contains the following stories (already estimated):

| Story | Points |
|-------|--------|
| User registration | 5 |
| Email verification | 3 |
| Profile editing | 8 |
| Password reset | 5 |
| Two-factor authentication | 13 |
| OAuth login (Google) | 8 |
| Account deletion | 3 |
| Admin user management | 13 |

(a) Which stories fit in Sprint 1 if the team selects in priority order from top to bottom?
(b) How many sprints are needed to complete all stories?
(c) If a senior engineer leaves mid-project, reducing velocity to 22 points/sprint, how does the release timeline change?

### Exercise 4: Identify Estimation Pitfalls

For each scenario, name the estimation bias or pitfall from Section 13, and propose a concrete mitigation technique.

(a) A developer estimates a new module will take "about a week" and their manager immediately says "Great, so we can ship it in five days."
(b) The team spends the first eight days of a ten-day sprint on design discussions, then rushes implementation in the last two days.
(c) A feature that was estimated at 5 story points consistently takes 3× longer than comparable 5-point features from other teams.
(d) A project is reported as "almost done" for six consecutive weeks while the same 20% of work remains.

### Exercise 5: Build a WBS and Find the Critical Path

You are managing a two-month data migration project. Decompose the work into a WBS with at least three levels and then list the key tasks with dependencies:

| Task | Duration | Predecessors |
|------|----------|--------------|
| Requirements analysis | 3 days | — |
| Source schema mapping | 4 days | Requirements analysis |
| Target schema design | 5 days | Requirements analysis |
| ETL script development | 8 days | Source schema mapping, Target schema design |
| Data validation rules | 3 days | Target schema design |
| Testing environment setup | 2 days | — |
| Migration dry run | 4 days | ETL script development, Testing environment setup |
| Validation testing | 3 days | Migration dry run, Data validation rules |
| Production migration | 1 day | Validation testing |

(a) Perform a forward and backward pass to compute ES, EF, LS, LF, and Float for each task.
(b) Identify the critical path and the project duration.
(c) If "ETL script development" is delayed by 3 days, what happens to the project end date?

---

**Previous**: [05. Software Modeling and UML](./05_Software_Modeling_and_UML.md) | **Next**: [07. Software Quality Assurance](./07_Software_Quality_Assurance.md)
