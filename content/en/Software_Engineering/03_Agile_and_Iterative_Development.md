# Lesson 3: Agile and Iterative Development

**Previous**: [Software Development Life Cycle](./02_Software_Development_Life_Cycle.md) | **Next**: [Requirements Engineering](./04_Requirements_Engineering.md)

---

Agile is not a methodology — it is a philosophy. The term describes a family of lightweight, iterative approaches to software development that prioritize responding to change, delivering working software frequently, and collaborating closely with customers. Agile emerged as a direct reaction to the failures of heavyweight, plan-driven processes for projects with uncertain or rapidly changing requirements.

**Difficulty**: ⭐⭐

**Prerequisites**:
- [Lesson 1: What Is Software Engineering](./01_What_Is_Software_Engineering.md)
- [Lesson 2: Software Development Life Cycle](./02_Software_Development_Life_Cycle.md)

**Learning Objectives**:
- Explain the four values and twelve principles of the Agile Manifesto
- Describe the Scrum framework: roles, events, and artifacts
- Apply Kanban principles and visualize work with a Kanban board
- Identify the core practices of Extreme Programming (XP)
- Explain Lean Software Development's seven principles
- Describe approaches to scaling agile (SAFe, LeSS, Spotify model)
- Calculate and interpret agile metrics: velocity, burndown, cycle time, lead time
- Choose between agile and plan-driven approaches based on project context
- Recognize common agile anti-patterns

---

## 1. The Agile Manifesto

On February 11–13, 2001, seventeen software practitioners gathered at the Snowbird ski resort in Utah. They represented a diverse set of lightweight methodologies: Extreme Programming, Scrum, DSDM, Adaptive Software Development, Crystal, Feature-Driven Development, and Pragmatic Programming. Despite their different approaches, they agreed on a common set of values and principles, which they published as the **Agile Manifesto**.

### The Four Values

The Manifesto states:

> We are uncovering better ways of developing software by doing it and helping others do it. Through this work we have come to value:

| We value... | Over... |
|------------|---------|
| **Individuals and interactions** | Processes and tools |
| **Working software** | Comprehensive documentation |
| **Customer collaboration** | Contract negotiation |
| **Responding to change** | Following a plan |

> *"That is, while there is value in the items on the right, we value the items on the left more."*

Crucially, the right-hand items are not worthless — processes, documentation, contracts, and plans all have value. The Manifesto is about *priorities* when trade-offs must be made.

### The Twelve Principles

The Manifesto is supported by twelve principles:

1. **Customer satisfaction through early and continuous delivery** of valuable software
2. **Welcome changing requirements**, even late in development. Agile processes harness change for the customer's competitive advantage
3. **Deliver working software frequently**, from a couple of weeks to a couple of months, with a preference for shorter timescales
4. **Business people and developers must work together** daily throughout the project
5. **Build projects around motivated individuals**. Give them the environment and support they need, and trust them to get the job done
6. **Face-to-face conversation** is the most efficient and effective method of conveying information
7. **Working software is the primary measure of progress**
8. Agile processes promote **sustainable development**. The sponsors, developers, and users should be able to maintain a constant pace indefinitely
9. **Continuous attention to technical excellence** and good design enhances agility
10. **Simplicity** — the art of maximizing the amount of work not done — is essential
11. The best architectures, requirements, and designs emerge from **self-organizing teams**
12. At regular intervals, the team reflects on how to become more effective, then **tunes and adjusts its behavior** accordingly

---

## 2. Scrum

Scrum is the most widely adopted agile framework. Developed by Ken Schwaber and Jeff Sutherland in the early 1990s and formalized in the 1995 OOPSLA paper, Scrum provides a structured but lightweight process for iterative development.

Scrum is defined by three pillars:
- **Transparency**: The process and work must be visible to all
- **Inspection**: Frequent inspection of artifacts and progress toward goals
- **Adaptation**: If something deviates from expected results, the process must be adjusted

### 2.1 Scrum Roles

**Product Owner (PO)**
- Owns the Product Backlog
- Responsible for maximizing the value of the product
- Represents stakeholders' interests to the development team
- Accepts or rejects completed work
- Single person, not a committee

**Scrum Master (SM)**
- Servant-leader for the team
- Facilitates Scrum events
- Removes impediments that block the team
- Coaches the team and organization on Scrum
- Not a project manager or team lead in the traditional sense

**Development Team**
- Cross-functional: contains all skills needed to deliver the product increment
- Self-organizing: decides how to accomplish the work (not told by others)
- 3–9 people (the "two-pizza rule")
- No sub-teams or hierarchies within the development team
- Collectively accountable for the increment

### 2.2 Scrum Events

**The Sprint**
The heartbeat of Scrum. A Sprint is a time-boxed iteration of 1–4 weeks (commonly 2 weeks) during which a potentially releasable increment of the product is created.

Rules:
- Duration is fixed (no extending Sprints)
- No changes that endanger the Sprint Goal
- Quality standards do not decrease
- Can be cancelled only by the Product Owner (rare)

```
Sprint Lifecycle:

Day 0               Day 1-N              Last Day
   │                    │                    │
   ▼                    ▼                    ▼
Sprint         Daily work             Sprint Review
Planning    ─────────────────────────► Sprint Retro
   │          (Daily Standup each day)      │
   │                                        │
   └──────────── Sprint Goal ───────────────┘
```

**Sprint Planning**
- Duration: Up to 8 hours for a 4-week Sprint (proportionally less for shorter Sprints)
- Outputs: Sprint Goal, Sprint Backlog
- Part 1: *What* can be done this Sprint? (PO presents top backlog items; team selects)
- Part 2: *How* will the work be done? (Team creates tasks and estimates)

**Daily Scrum (Daily Standup)**
- Duration: 15 minutes, every day
- Format: Each team member addresses (traditionally):
  - What did I do yesterday that helped meet the Sprint Goal?
  - What will I do today to help meet the Sprint Goal?
  - Do I see any impediments?
- Purpose: Inspect progress toward Sprint Goal and adapt the Sprint Backlog
- Not a status report to management — it is the team coordinating with itself

**Sprint Review**
- Duration: Up to 4 hours for a 4-week Sprint
- The team demonstrates the completed increment to stakeholders
- Stakeholders provide feedback; PO updates the backlog based on that feedback
- Informal, collaborative, not a gate or approval meeting

**Sprint Retrospective**
- Duration: Up to 3 hours for a 4-week Sprint
- The team reflects on the *process* (not the product)
- Identifies what went well, what didn't, and what to improve next Sprint
- Key output: One or two actionable improvement items for the next Sprint

### 2.3 Scrum Artifacts

**Product Backlog**
- An ordered list of everything that might be needed in the product
- Single source of requirements for any changes
- Owned and managed by the Product Owner
- Items are called Product Backlog Items (PBIs) or User Stories
- Higher-priority items are more refined (smaller, with clear acceptance criteria); lower-priority items are coarser
- Never "complete" — evolves as the product and environment change

Example Product Backlog:

```
Priority  │ Story                                        │ Estimate
──────────┼──────────────────────────────────────────────┼──────────
1 (High)  │ User can log in with email/password          │ 3 pts
2         │ User receives email confirmation on signup   │ 2 pts
3         │ Admin can view all registered users          │ 5 pts
4         │ User can reset forgotten password            │ 3 pts
5 (Low)   │ Support OAuth login with Google              │ 8 pts
...       │ ...                                          │ ...
```

**Sprint Backlog**
- The subset of Product Backlog items selected for the Sprint, plus a plan for delivering them
- Owned by the Development Team
- Visible to all; updated daily
- The team adds, modifies, and removes tasks as they learn during the Sprint

**Increment**
- The sum of all completed Product Backlog items at the end of a Sprint, plus the value of all previous Sprints
- Must meet the team's **Definition of Done (DoD)** to count as an Increment
- Must be "potentially shippable" — functional, tested, integrated

**Definition of Done (DoD)**
A shared, explicit understanding of what "complete" means. A typical DoD might include:
- Code reviewed by at least one peer
- Unit tests written and passing
- Integration tests passing
- No known critical defects
- Documentation updated
- Performance within acceptable bounds
- Deployed to staging environment

---

## 3. Kanban

Kanban is a lean method for managing and improving workflow. Originally developed by Toyota for manufacturing (the term is Japanese for "visual card" or "signboard"), it was adapted for software development by David Anderson around 2007.

### Core Kanban Principles

1. **Start with what you do now**: Kanban does not prescribe a specific process; it layers improvement onto your existing process
2. **Agree to pursue incremental, evolutionary change**: Do not attempt radical change; improve gradually
3. **Respect the current process, roles, and responsibilities**: Do not assume the current process is broken
4. **Encourage acts of leadership at all levels**: Everyone on the team can suggest improvements

### Core Kanban Practices

1. **Visualize the workflow**: Make work visible on a Kanban board
2. **Limit Work in Progress (WIP)**: Set explicit limits on how many items can be in each stage simultaneously
3. **Manage flow**: Monitor and optimize how work flows through the system
4. **Make policies explicit**: Everyone understands how decisions are made
5. **Implement feedback loops**: Regular meetings to review flow and quality
6. **Improve collaboratively, evolve experimentally**: Use metrics to guide improvement

### Kanban Board

```
┌──────────┬──────────────┬──────────────┬──────────┬──────────┐
│ Backlog  │   Analysis   │ Development  │ Testing  │   Done   │
│          │  (WIP: ≤ 2)  │  (WIP: ≤ 3)  │ (WIP:≤2) │          │
├──────────┼──────────────┼──────────────┼──────────┼──────────┤
│          │              │              │          │          │
│ Story A  │ Story D      │ Story E      │ Story G  │ Story I  │
│          │              │              │          │          │
│ Story B  │ Story F      │ Story H      │          │ Story J  │
│          │              │              │          │          │
│ Story C  │              │ Story K      │          │ Story L  │
│          │              │              │          │          │
│ Story M  │              │              │          │          │
└──────────┴──────────────┴──────────────┴──────────┴──────────┘
```

### WIP Limits

WIP limits are central to Kanban. They prevent the system from being overloaded and force the team to finish work before starting new work.

Benefits of WIP limits:
- **Reduce multitasking**: Context switching is costly; WIP limits encourage focus
- **Surface bottlenecks**: When a column is at its WIP limit, blockages become visible
- **Reduce lead time**: Little's Law: $L = \lambda W$ (Lead time = WIP / Throughput rate)
- **Improve quality**: Unfinished work in progress accumulates defects

### Kanban vs. Scrum

| Dimension | Scrum | Kanban |
|-----------|-------|--------|
| Iteration | Fixed-length Sprints | Continuous flow |
| Roles | PO, SM, Dev Team | No prescribed roles |
| WIP limits | Implicit (Sprint scope) | Explicit per stage |
| Change | No changes within a Sprint | Can change anytime |
| Velocity metric | Story points per Sprint | Lead time, cycle time, throughput |
| Best for | New product development | Support, operations, maintenance |

---

## 4. Extreme Programming (XP)

Extreme Programming (XP), created by Kent Beck and described in *Extreme Programming Explained* (1999), is an agile framework that takes proven software development practices to extreme levels. If code review is good, review all the time (pair programming). If testing is good, test all the time (TDD). If integration is good, integrate continuously.

### XP's Core Values

- **Communication**: Problems arise from lack of communication; XP promotes constant communication
- **Simplicity**: Do the simplest thing that could possibly work today
- **Feedback**: Fast feedback at every level — from tests, customers, and team members
- **Courage**: Make hard decisions: refactor mercilessly, tell customers bad news, discard failing code
- **Respect**: Team members respect each other and the work

### XP Practices

**Planning**
- *Release planning*: Customer defines stories; team estimates; together they define release scope
- *Iteration planning*: 1–3 week iterations planned by the team
- *Small releases*: Deliver working software every 1–3 weeks

**Design**
- *Simple design*: Implement only what is needed now; no speculative features (YAGNI: You Ain't Gonna Need It)
- *System metaphor*: A shared story describing how the system works as a whole
- *Refactoring*: Continuously improve the design of existing code

**Coding**
- *Pair programming*: Two developers work at one keyboard at all times
  - Driver: writes the code
  - Navigator: reviews in real time, thinks about the bigger picture
  - Pairs rotate frequently
- *Collective ownership*: Anyone can improve any part of the code at any time; there is no individual ownership
- *Coding standards*: Shared conventions enable collective ownership

**Testing**
- *Test-Driven Development (TDD)*: Write a failing test before writing code; make it pass; refactor
  ```
  Red → Green → Refactor cycle:
  1. Write a failing test (Red)
  2. Write just enough code to pass the test (Green)
  3. Refactor the code and test (Refactor)
  4. Repeat
  ```
- *Customer tests*: Customers define acceptance tests for each story

**Integration**
- *Continuous integration (CI)*: Every developer integrates their changes with the mainline multiple times per day; automated tests run on every integration
- *On-site customer*: A real customer is available full-time to answer questions and make decisions

### XP vs. Scrum

XP is more prescriptive about *engineering practices* (TDD, pair programming, CI); Scrum is more prescriptive about *process* (Sprints, Daily Scrum, Review, Retrospective). Many teams use Scrum for process management and adopt XP practices for engineering quality. This combination is sometimes called "Scrum-ban" or "Disciplined Agile."

---

## 5. Lean Software Development

Lean Software Development, introduced by Mary and Tom Poppendieck in *Lean Software Development: An Agile Toolkit* (2003), adapts Toyota's Lean Manufacturing principles to software.

### The Seven Lean Principles

| # | Principle | Software Interpretation |
|---|-----------|------------------------|
| 1 | **Eliminate Waste** | Remove anything that does not add value: unnecessary features, waiting time, handoffs, defects |
| 2 | **Build Quality In** | Find defects at the source (TDD, pair review); do not rely on testing to find defects after the fact |
| 3 | **Create Knowledge** | Software development is a knowledge-creation activity; invest in learning and experimentation |
| 4 | **Defer Commitment** | Make decisions at the last responsible moment, not upfront; keep options open as long as practical |
| 5 | **Deliver Fast** | Speed enables learning and reduces waste; a fast cycle reveals problems sooner |
| 6 | **Respect People** | Empower teams; respect their expertise; create a culture of psychological safety |
| 7 | **Optimize the Whole** | Optimize the entire value stream, not local efficiencies; avoid suboptimization |

### Waste in Software Development

Lean identifies seven types of waste (*muda*) in manufacturing, which translate to software as:

| Manufacturing Waste | Software Equivalent |
|--------------------|---------------------|
| Inventory | Partially done work, undeployed features |
| Overproduction | Building features not yet needed |
| Extra processing | Unnecessary documentation, gold-plating |
| Transportation | Handoffs between teams without knowledge transfer |
| Waiting | Waiting for approvals, environments, decisions |
| Motion | Task switching, context switching |
| Defects | Bugs that must be found, reported, fixed |

---

## 6. Scaling Agile

Agile was conceived for small, co-located teams of 5–12 people. As organizations adopted agile at scale — dozens or hundreds of teams working on a single product — new frameworks emerged to coordinate this larger effort.

### 6.1 SAFe (Scaled Agile Framework)

SAFe, created by Dean Leffingwell, is the most widely adopted scaling framework. It organizes work into four levels:

```
Portfolio Level:    Strategic themes, epics, portfolio backlog
        │
        ▼
Large Solution:     Solution train (for very large systems)
        │
        ▼
Program Level:      Agile Release Train (ART), PI Planning
        │
        ▼
Team Level:         Individual Scrum/Kanban teams
```

Key SAFe concepts:
- **Agile Release Train (ART)**: 5–12 agile teams working together; synchronized on a 10-week Program Increment (PI)
- **PI Planning**: All teams gather for a 2-day event to plan the next PI together; creates cross-team alignment
- **Program Backlog**: Features and enablers planned at the program level

SAFe is comprehensive but heavyweight; it requires significant organizational investment to implement.

### 6.2 LeSS (Large-Scale Scrum)

LeSS, developed by Craig Larman and Bas Vodde, scales Scrum with minimal process additions. The philosophy: do as little as possible on top of Scrum.

```
LeSS (2-8 teams):
- One Product Owner, one Product Backlog
- One Sprint for all teams simultaneously
- One overall Sprint Review
- Each team has its own Daily Scrum and Retrospective
- Overall Retrospective for cross-team concerns

LeSS Huge (8+ teams):
- Introduce Area Product Owners for large requirement areas
- Otherwise, same structure
```

LeSS requires that teams are cross-functional and can each deliver a complete, integrated product increment.

### 6.3 Spotify Model

Not a formal framework, but an influential organizational design described by Henrik Kniberg and Anders Ivarsson in 2012 based on Spotify's structure.

```
Squad: Autonomous mini-startup; owns a feature area end-to-end (~8 people)
   │
   ├── Tribe: Collection of squads working in related areas (~10 squads)
   │
   ├── Chapter: Horizontal guild of specialists across squads (e.g., all iOS devs)
   │
   └── Guild: Communities of practice across chapters and tribes
```

Key ideas:
- Squads are autonomous: they choose their own tools and processes
- Chapters maintain technical standards and career development
- The model emphasizes culture over process

Note: The Spotify model is widely cited but also widely misunderstood. Spotify itself has moved on from this exact structure.

---

## 7. Agile Metrics

Agile teams use metrics to track progress, forecast delivery, and improve their process.

### 7.1 Velocity

**Velocity** is the amount of work (in story points) a team completes per Sprint.

```
Sprint 1:  Completed 21 points
Sprint 2:  Completed 18 points
Sprint 3:  Completed 23 points
Sprint 4:  Completed 20 points
Average velocity: (21+18+23+20) / 4 = 20.5 points/Sprint
```

Usage: Forecast how many Sprints are needed to deliver a backlog of N points.

Caution: Velocity is a *planning* tool, not a *performance* metric. Comparing velocity across teams is meaningless. "Velocity inflation" (gaming points) is a common anti-pattern.

### 7.2 Burndown Chart

A **Sprint Burndown Chart** shows the remaining work in the Sprint over time.

```
Story Points Remaining
30 │ ×
   │  ×
25 │   ×    ← Ideal line
   │    ×  /
20 │─────────── ideal
   │      × /
15 │       ×
   │        ×
10 │         ×
   │          ×
 5 │           ×
   │            ×
 0 └─────────────────────
   Day 1  5   10   Sprint End
```

If the actual line is above the ideal line, the team is behind. If below, they are ahead.

### 7.3 Cumulative Flow Diagram (CFD)

A CFD shows the number of items in each workflow stage over time. It is particularly useful for Kanban teams.

```
Items
^
│          ████ Done
│        ████████
│      ████████████ Testing
│    ████████████████
│  ████████████████████ Dev
│████████████████████████ Analysis
└──────────────────────────────► Time
```

A widening band in any stage indicates a bottleneck.

### 7.4 Cycle Time and Lead Time

- **Lead time**: Time from when a request enters the backlog to when it is delivered to the customer
- **Cycle time**: Time from when work *starts* on an item to when it is delivered

$\text{Lead Time} = \text{Wait Time} + \text{Cycle Time}$

Shorter cycle times mean faster feedback. Teams should track the distribution (histogram) of cycle times, not just the average. Outliers (very long cycle times) indicate systemic problems.

---

## 8. Agile vs. Waterfall: When to Choose Each

Both agile and waterfall have legitimate use cases. The choice should be driven by project characteristics:

| Factor | Prefer Agile | Prefer Waterfall/V-Model |
|--------|-------------|--------------------------|
| Requirements certainty | Unclear, evolving | Well-defined, stable |
| Customer involvement | Customer available, engaged | Limited customer availability |
| Risk profile | High risk of building wrong thing | High risk of integration failure |
| Team experience | Experienced, self-organizing | Less experienced, needs structure |
| Delivery pressure | Need value delivered early | Complete system needed at once |
| Regulatory context | Low to medium | High (FDA, DO-178C, ISO 26262) |
| Contract type | Time-and-materials, outcomes-based | Fixed-price, fixed-scope |
| Innovation level | Exploratory, new product | Well-understood domain |

---

## 9. Common Agile Anti-Patterns

Adopting agile ceremonies without the agile mindset produces "zombie agile" — the form without the function.

### Anti-Patterns by Area

**Planning**
- *Wagile*: Doing agile ceremonies but with a fixed upfront plan; changes are not actually welcome
- *Sprint overloading*: Consistently pulling more into a Sprint than can be completed
- *No Product Owner*: Product decisions made by committee or by developers

**Daily Standup**
- *Status reports*: Treating the standup as a reporting meeting to managers rather than team coordination
- *Problem-solving in standup*: Turning the standup into a design meeting; long discussions that don't involve everyone
- *Not standing*: Multi-hour seated "standups"

**Backlog**
- *Story hoarding*: Product Backlog has hundreds of items, most of which will never be worked on
- *Story factories*: Teams write stories but rarely complete and deliver them
- *No refinement*: Sprint Planning becomes chaotic because stories are not ready

**Retrospectives**
- *Blame sessions*: Retrospectives become opportunities to criticize individuals
- *No follow-through*: Action items from retrospectives are not acted upon in the next Sprint
- *Cancelled retrospectives*: "We don't have time" — the team never improves its process

**Velocity**
- *Velocity as a performance target*: Management sets velocity targets; teams inflate estimates to meet them
- *Comparing team velocities*: Velocity is calibrated per-team; comparison is meaningless and harmful

**Technical**
- *No Definition of Done*: "Done" means different things to different people; quality is inconsistent
- *Skipping technical practices*: Adopting Scrum ceremonies but ignoring TDD, refactoring, CI/CD
- *Technical debt accumulation*: Cutting corners each Sprint without allocating time for improvement

---

## Summary

Agile represents a fundamental shift in how software development is approached: from detailed upfront planning toward embracing uncertainty and adapting continuously. The Agile Manifesto's four values and twelve principles provide the philosophical foundation; Scrum, Kanban, XP, and Lean provide specific practices.

Key takeaways:
- The **Agile Manifesto** values individuals, working software, customer collaboration, and responding to change over their alternatives
- **Scrum** provides a structured iterative process with defined roles (PO, SM, Dev Team), events (Sprint, Planning, Standup, Review, Retro), and artifacts (Product Backlog, Sprint Backlog, Increment)
- **Kanban** manages continuous flow with explicit WIP limits; ideal for maintenance and support contexts
- **XP** takes proven engineering practices to the extreme: TDD, pair programming, continuous integration
- **Lean** focuses on eliminating waste throughout the value stream
- Scaling agile requires additional frameworks (SAFe, LeSS, Spotify model) with increasing coordination overhead
- Agile metrics — velocity, burndown, cycle time, lead time — measure throughput and flow, not individual performance
- The choice between agile and plan-driven methods depends on requirements stability, customer availability, regulatory context, and project risk
- Agile anti-patterns arise when teams adopt ceremonies without the underlying values and technical discipline

---

## Practice Exercises

**Exercise 1**: You are the Scrum Master for a new 5-person team building a mobile banking app. The Sprint length is 2 weeks. Write a detailed agenda for the first Sprint Planning meeting, including: how you will introduce the Product Backlog, how the team will select stories, and how they will create their task plan. Estimate time allocations for each section.

**Exercise 2**: Design a Kanban board for a 3-person team that handles both new feature development and production support tickets. Specify columns, WIP limits for each column, and explain how you chose the WIP limits. Describe how the team should handle urgent production incidents that arrive mid-flow.

**Exercise 3**: Your team's last four Sprints had velocities of 22, 19, 25, and 21 points. The Product Backlog has 180 points remaining. Estimate when the project will complete. Now the Product Owner wants to add a new epic of 45 points with high priority. How does this change your forecast? What information would you need to give the Product Owner a reliable commitment?

**Exercise 4**: Identify which of the twelve Agile Manifesto principles are violated by each of the following anti-patterns: (a) Management asks developers to report Sprint progress in a weekly status meeting. (b) The team skips the Sprint Retrospective because they are behind schedule. (c) The Product Owner is unavailable and delegates to a business analyst who cannot make decisions.

**Exercise 5**: A large insurance company has 50 development teams working on a single enterprise policy management platform. Compare SAFe and LeSS as scaling approaches for this context. What are the key trade-offs? Which would you recommend, and what organizational changes would be required?

---

## Further Reading

- **Agile Manifesto**: https://agilemanifesto.org — Read the original values and principles
- **Ken Schwaber & Jeff Sutherland**, *The Scrum Guide* (2020): https://scrumguides.org — The official, free definition of Scrum
- **David J. Anderson**, *Kanban: Successful Evolutionary Change for Your Technology Business* (2010) — The foundational Kanban text
- **Kent Beck**, *Extreme Programming Explained: Embrace Change* (2nd ed., 2004) — XP's original text
- **Mary Poppendieck & Tom Poppendieck**, *Lean Software Development: An Agile Toolkit* (2003)
- **Henrik Kniberg**, *Scrum and XP from the Trenches* (free PDF) — Practical Scrum implementation guide
- **Dean Leffingwell**, *SAFe 5.0 Distilled* (2020) — Scaled Agile Framework guide
- **Craig Larman & Bas Vodde**, *Large-Scale Scrum: More with LeSS* (2016)
- **Mike Cohn**, *Agile Estimating and Planning* (2005) — Comprehensive coverage of story points, velocity, and planning

---

**Previous**: [Software Development Life Cycle](./02_Software_Development_Life_Cycle.md) | **Next**: [Requirements Engineering](./04_Requirements_Engineering.md)
