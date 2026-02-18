# Lesson 11: Software Maintenance and Evolution

**Previous**: [Project Management](./10_Project_Management.md) | **Next**: [Process Improvement](./12_Process_Improvement.md)

Software does not wear out the way physical machinery does. A database binary stored on a server experiences no friction, corrosion, or fatigue. Yet software still "ages" — not because its bits degrade, but because the world around it changes. Operating systems evolve, security vulnerabilities are discovered, business rules shift, and user expectations rise. Understanding how to sustain and evolve software after its initial release is one of the most economically significant skills in software engineering.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Software Development Life Cycle concepts (Lesson 02)
- Basic software design principles
- Familiarity with version control (Git)

**Learning Objectives**:
- Distinguish the four types of software maintenance and know when each applies
- Explain Lehman's Laws and their implications for long-lived systems
- Evaluate strategies for modernizing legacy systems
- Apply refactoring techniques to improve code without changing behavior
- Understand reverse engineering, software aging, and migration strategies
- Select appropriate deprecation policies for APIs and services
- Measure maintenance health using key metrics

---

## 1. The Scale of Software Maintenance

Software maintenance is the modification of a software product after delivery to correct faults, improve performance, or adapt it to a changed environment (ISO 14764). It is not a secondary concern — it dominates the software lifecycle.

**Typical cost distribution across the software lifecycle:**

| Phase | Approximate Share of Total Lifetime Cost |
|---|---|
| Requirements and design | 5–10% |
| Coding and unit testing | 10–15% |
| Integration and system testing | 10–15% |
| **Maintenance** | **60–80%** |

Several factors drive this disproportion:
- Systems that succeed grow in users, features, and complexity
- Personnel turnover means maintainers often did not write the original code
- Requirements continue evolving as the business evolves
- Technology platforms change beneath the application

---

## 2. Types of Software Maintenance (ISO 14764)

ISO 14764 defines four categories of maintenance. Understanding which type of work you are doing helps allocate resources correctly and set appropriate expectations.

### 2.1 Corrective Maintenance

**Goal**: Fix defects that cause incorrect behavior or system failures.

Corrective maintenance addresses bugs reported by users or detected by monitoring. It ranges from trivial one-line typo fixes to multi-week investigations of race conditions in distributed systems.

```python
# Bug reported: Division by zero when user has no orders
# Before (buggy)
def average_order_value(user_id: int) -> float:
    orders = get_orders(user_id)
    total = sum(o.value for o in orders)
    return total / len(orders)  # ZeroDivisionError if no orders

# After (corrective fix)
def average_order_value(user_id: int) -> float:
    orders = get_orders(user_id)
    if not orders:
        return 0.0
    total = sum(o.value for o in orders)
    return total / len(orders)
```

### 2.2 Adaptive Maintenance

**Goal**: Modify the software to work in a changed environment.

The environment includes operating systems, hardware, database engines, third-party APIs, legal regulations, and company policies. None of these are stable forever.

Examples:
- Updating TLS certificate handling after TLS 1.0 is deprecated
- Migrating from Python 2 to Python 3
- Adapting to GDPR requirements introduced after initial deployment
- Switching from a deprecated payment gateway to a new provider

### 2.3 Perfective Maintenance

**Goal**: Improve performance, maintainability, or add new features requested by users.

This is often the most resource-intensive category. As the system proves its value, stakeholders request enhancements. The line between perfective maintenance and new development is blurry — the key distinction is that the system already exists and is in production.

### 2.4 Preventive Maintenance

**Goal**: Reduce future failures by improving reliability, maintainability, or security before problems occur.

Examples:
- Refactoring a tangled module to reduce its cyclomatic complexity
- Adding automated tests to an untested critical path
- Rotating encryption keys before they expire
- Upgrading a dependency before its known vulnerability becomes actively exploited

| Type | Trigger | Example |
|---|---|---|
| Corrective | Bug report or failure | Fix null pointer exception in production |
| Adaptive | Environmental change | Migrate to Python 3.12 |
| Perfective | User or business request | Add CSV export to reports |
| Preventive | Proactive analysis | Refactor 2,000-line monolithic function |

In practice, most maintenance backlogs contain a mix of all four. Purely corrective shops are reactive and fragile; investing in preventive maintenance is the hallmark of a mature engineering organization.

---

## 3. Lehman's Laws of Software Evolution

Meir "Manny" Lehman and László Bélády studied large software systems at IBM in the 1970s–1990s and derived empirical laws that hold remarkably well across decades of software history.

| Law | Name | Statement |
|---|---|---|
| I | **Continuing Change** | An E-type system must be continually adapted or it becomes progressively less satisfactory |
| II | **Increasing Complexity** | As a system evolves, its complexity increases unless work is done to reduce it |
| III | **Self Regulation** | Global system attributes (size, activity rates) are self-regulating and statistically invariant |
| IV | **Conservation of Organizational Stability** | The average effective global activity rate in an evolving system is invariant |
| V | **Conservation of Familiarity** | The content of successive releases is statistically invariant |
| VI | **Continuing Growth** | Functional content must increase to maintain user satisfaction over its lifetime |
| VII | **Declining Quality** | Quality will appear to decline unless rigorously maintained and adapted to operational environment changes |
| VIII | **Feedback System** | E-type evolution processes are multi-level, multi-loop, multi-agent feedback systems |

**Practical implications:**

- **Law I**: If your organization stops investing in a successful system, it will erode. "Freeze and forget" is not a stable strategy.
- **Law II**: Complexity is the natural direction of entropy. Preventive refactoring, architectural reviews, and technical debt reduction are essential counter-pressures.
- **Law VII**: Declining quality is the default trajectory, not the exception. Quality requires active, sustained investment.

---

## 4. Legacy Systems

A **legacy system** is a system that is critical to the business but difficult to change due to age, technology, design, or documentation gaps.

### 4.1 Characteristics of Legacy Systems

- Built with obsolete technology (COBOL, Visual Basic 6, ASP classic)
- Little or no automated test coverage
- Poor or missing documentation
- High coupling between components, making changes risky
- Known only by long-tenured staff (or staff who have left)
- High business value — the system actually works and is trusted

The last point is critical. Legacy systems are often mocked, but they represent decades of encoded business logic. The mainframe payroll system running a bank may be 40 years old, but it processes billions of dollars correctly every day.

### 4.2 Modernization Strategies

| Strategy | Description | Risk | Cost |
|---|---|---|---|
| **Leave it alone** | Accept the status quo | Accumulating | Low now, high later |
| **Wrap it** | Add a modern API layer around the legacy core | Low | Medium |
| **Extend it** | Add new capabilities without touching core | Medium | Medium |
| **Rewrite** | Build a replacement from scratch | High | Very high |
| **Replace** | Buy a COTS package to replace it | Medium | High |
| **Migrate** | Move to modern platform incrementally | Low–Medium | Medium–High |

The **Strangler Fig Pattern** (popularized by Martin Fowler) is the preferred incremental approach. Named after a vine that grows around a tree and eventually replaces it:

```
Phase 1: Intercept         Phase 2: Redirect         Phase 3: Strangle

  Requests                   Requests                  Requests
     │                          │                         │
     ▼                          ▼                         ▼
  ┌──────┐                  ┌──────┐                  ┌──────────┐
  │Legacy│                  │Router│                  │New System│
  │System│                  └──┬───┘                  └──────────┘
  └──────┘               Legacy│  New                 (Legacy retired)
                          ┌────┘  └──────┐
                       ┌──────┐      ┌──────────┐
                       │Legacy│      │New System│
                       │(some)│      │(growing) │
                       └──────┘      └──────────┘
```

A strangler fig migration allows:
- No big-bang cutover (which is the single highest-risk event in any migration)
- Running old and new systems in parallel for validation
- Incremental delivery of value
- The ability to pause or reverse if problems arise

---

## 5. Refactoring vs. Rewriting

**Refactoring** is changing the internal structure of code to improve its design without changing its observable external behavior. It is disciplined, incremental, and safe when backed by tests.

**Rewriting** is discarding existing code and building from scratch. It is occasionally necessary but is usually riskier and more expensive than teams expect.

### 5.1 The Case Against Rewriting

Joel Spolsky's famous article "Things You Should Never Do, Part I" (2000) argues that rewriting from scratch is almost always a mistake. Reasons:

- The "ugly" old code contains thousands of bug fixes that are not in any specification
- Rewrites take 3–5× longer than estimated
- By the time the rewrite ships, the original system has continued evolving, so the rewrite is already behind
- The new team makes different (not necessarily better) design mistakes

### 5.2 When Rewriting May Be Justified

- The original technology platform is end-of-life with no migration path
- The codebase has zero test coverage and is too entangled to safely refactor
- The domain model is fundamentally wrong and prevents necessary features
- Business requirements have changed so completely that the current design is an obstacle, not an asset

### 5.3 Common Refactoring Patterns

```python
# Extract Method: decompose a long method into named sub-functions
# Before
def process_order(order):
    # Validate
    if order.quantity <= 0:
        raise ValueError("Quantity must be positive")
    if order.product_id not in get_valid_products():
        raise ValueError("Invalid product")
    # Calculate price
    base_price = get_price(order.product_id)
    discount = 0.1 if order.quantity > 100 else 0
    total = base_price * order.quantity * (1 - discount)
    # Save
    db.save(Order(order.product_id, order.quantity, total))
    return total

# After (refactored)
def process_order(order):
    validate_order(order)
    total = calculate_total(order)
    save_order(order, total)
    return total

def validate_order(order):
    if order.quantity <= 0:
        raise ValueError("Quantity must be positive")
    if order.product_id not in get_valid_products():
        raise ValueError("Invalid product")

def calculate_total(order) -> float:
    base_price = get_price(order.product_id)
    discount = 0.1 if order.quantity > 100 else 0
    return base_price * order.quantity * (1 - discount)

def save_order(order, total: float):
    db.save(Order(order.product_id, order.quantity, total))
```

**The Refactoring Rule**: Never refactor without a test suite that confirms behavior is preserved.

---

## 6. Reverse Engineering and Program Comprehension

When maintaining code you did not write (or wrote years ago and have since forgotten), you must perform **program comprehension** — understanding what the code does, why it does it that way, and how changes will propagate.

### 6.1 Reverse Engineering Techniques

- **Static analysis**: Reading code, call graphs, dependency maps without executing it
- **Dynamic analysis**: Running the system and observing behavior (profilers, debuggers, log analysis)
- **Documentation recovery**: Mining commit history, issue trackers, old emails for design rationale
- **Concept location**: Identifying which code module implements which business concept

### 6.2 Tools for Program Comprehension

| Category | Tools |
|---|---|
| Static analysis | SonarQube, ESLint, PyLint, Understand |
| Call graph generation | Doxygen, pycallgraph, CodeScene |
| Dependency visualization | Sourcegraph, IntelliJ dependency view |
| Runtime profiling | py-spy, perf, async-profiler |
| Git history mining | `git log --follow`, git blame, GitLens |

---

## 7. Software Aging and Rejuvenation

### 7.1 Software Aging

David Parnas introduced the concept of **software aging** in 1994. Software ages because:

1. **Drift**: The system's design assumptions become less valid over time as the environment and requirements change
2. **Accumulation**: Each patch and quick fix leaves behind complexity, unused code, and inconsistency
3. **Ignorance**: Knowledge of why things are the way they are gradually leaves the team

Symptoms of software aging:
- Increasing time to implement new features
- High defect rates in old modules
- Fear of touching certain parts of the codebase
- Only one or two people understand specific subsystems
- Build and test cycles grow progressively longer

### 7.2 Software Rejuvenation

**Rejuvenation** is the planned, sustained effort to reverse software aging:

- Regular refactoring sprints or "tech debt sprints"
- Architectural reviews and modernization roadmaps
- Knowledge transfer through pair programming and documentation
- Replacing outdated frameworks and libraries on a planned schedule
- Improving build and test infrastructure to reduce feedback latency

A useful heuristic: allocate 10–20% of engineering capacity to ongoing rejuvenation. Teams that skip this eventually face a "big rewrite" crisis.

---

## 8. Technical Debt

Ward Cunningham coined the term **technical debt** in 1992 to describe the implied cost of rework caused by choosing an easy solution now instead of a better approach that would take longer. Like financial debt, technical debt accumulates interest over time — every future change in a debt-laden area costs more than it otherwise would.

### 8.1 Types of Technical Debt

Martin Fowler's **Technical Debt Quadrant** classifies debt by intent and prudence:

```
                  Reckless           Prudent
                     │                 │
Deliberate    "We don't have    "We must ship now
              time for design"  and deal with it later"
                     │                 │
─────────────────────┼─────────────────┼─────────────
                     │                 │
Inadvertent   "What's layering?"  "Now we know how
                     │            we should have done it"
```

| Quadrant | Example | Appropriate Response |
|---|---|---|
| Deliberate + Reckless | Skipping design "to go fast" | Stop — this rarely pays off |
| Deliberate + Prudent | Hardcoding config to hit a launch deadline | Accept, with a tracked payback plan |
| Inadvertent + Reckless | Design antipatterns from inexperience | Invest in team skills, fix during maintenance |
| Inadvertent + Prudent | Realizing a better design after implementation | Normal learning; refactor when economically justified |

### 8.2 Technical Debt Register

Similar to a risk register, a **technical debt register** makes debt visible and actionable:

```markdown
| ID   | Location          | Description                          | Impact     | Effort  | Priority |
|------|-------------------|--------------------------------------|------------|---------|----------|
| TD01 | auth/session.py   | Pickle serialization; blocks upgrades | High       | 3 days  | P1       |
| TD02 | reports/generate  | 2,400-line god function              | High       | 5 days  | P1       |
| TD03 | models/product.py | No type annotations                  | Medium     | 2 days  | P2       |
| TD04 | tests/integration | Flaky sleep()-based async tests      | Medium     | 3 days  | P2       |
| TD05 | frontend/styles   | 800-line CSS file, no BEM            | Low        | 4 days  | P3       |
```

### 8.3 Managing Technical Debt

Effective debt management balances paydown against feature delivery:

- **The Boy Scout Rule**: "Always leave the code a little cleaner than you found it." Small, continuous improvements compound over time.
- **Dedicated debt sprints**: Allocate one sprint per quarter exclusively to debt reduction. Teams that never do this eventually reach a crisis point.
- **Definition of Done**: Include debt creation criteria in the team's Definition of Done (e.g., "No new functions longer than 50 lines without review").
- **Debt budgets**: Some teams set a maximum debt-to-feature ratio — for every 4 sprints of feature work, at least 1 sprint of debt reduction.

```python
# Code review checklist item: flag new technical debt
# Good practice: track it in the PR description

## PR #847: Add bulk export feature

### Technical Debt Created
- The CSV serialization is hardcoded for the current schema.
  Tracked as TD-023 in tech debt register.
  Plan to abstract with a serializer interface in Q3.

### Technical Debt Reduced
- Refactored the `ReportBuilder` class (was 800 lines → now 4 classes × ~200 lines)
  Closes TD-019.
```

### 8.4 Debt-Induced Velocity Decline

Without active debt management, teams experience a characteristic pattern:

```
Team Velocity
     │
100% │████████████
 90% │████████████████
 80% │████████████████████
 70% │████████████████████████▌
 60% │████████████████████████████
 50% │████████████████████████████████
     └────────────────────────────────────▶ Time (quarters)
           Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8

Without debt reduction, velocity declines ~10% per quarter in high-churn codebases.
```

This pattern explains why a team that was delivering 40 story points per sprint in year one is delivering 18 story points in year three — not because the team is less capable, but because each change now requires navigating a complex web of accumulated technical debt.

---

## 10. Migration Strategies

When moving from one platform, database, or architecture to another, three main strategies exist:

### 8.1 Big Bang Migration

Replace the old system with the new one in a single cutover event.

```
Old System  ──────────────────────────────── Cutover ──────── Retired
New System                                           ────────────────
```

- **Advantage**: Simple to manage; no long-running parallel operation
- **Risk**: If the new system has defects on day one, the full user population is impacted immediately
- **When appropriate**: Small systems, low traffic, well-understood domains with excellent test coverage

### 8.2 Phased Migration

Migrate functionality module by module, running old and new systems in parallel.

```
Module A:  Old ──── New
Module B:        Old ──── New
Module C:              Old ──── New
```

- **Advantage**: Lower risk per phase; allows learning before migrating complex parts
- **Risk**: Dual-system data synchronization complexity; longer overall timeline
- **When appropriate**: Most medium-to-large systems (the default recommendation)

### 8.3 Parallel Operation

Run both old and new systems simultaneously, comparing outputs.

```
All Requests ──┬──▶ Old System ──▶ Authoritative Response
               └──▶ New System ──▶ Logged for comparison
```

- **Advantage**: Validate correctness before switching traffic
- **Risk**: Double the infrastructure cost; complex comparison logic
- **When appropriate**: High-stakes systems (financial calculations, medical records)

---

## 11. Deprecation Policies

Deprecation is the process of phasing out APIs, features, or services. Well-managed deprecation prevents breaking changes from surprising consumers.

### 9.1 Deprecation Best Practices

```python
import warnings

def legacy_calculate_discount(order_total: float, user_tier: str) -> float:
    """
    Calculate discount based on order total and user tier.

    .. deprecated:: 2.4.0
        Use :func:`calculate_discount_v2` instead. This function will be
        removed in version 4.0.0 (scheduled for 2026-Q1).
    """
    warnings.warn(
        "legacy_calculate_discount is deprecated and will be removed in v4.0.0. "
        "Use calculate_discount_v2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return order_total * 0.1 if user_tier == "premium" else 0.0
```

**Deprecation timeline guidelines:**

| System Type | Minimum Deprecation Period |
|---|---|
| Internal library (single team) | 1 sprint to 1 quarter |
| Internal API (multiple teams) | 2–4 quarters |
| Public API (external developers) | 1–2 years minimum |
| Platform/language feature | 2–5 years |

Communicate deprecations through: changelogs, compiler/runtime warnings, developer newsletters, migration guides.

---

## 12. Anti-Patterns in Software Maintenance

Recognizing common failure modes helps teams avoid them proactively.

### 12.1 The Lava Layer Anti-Pattern

Each generation of developers adds new layers on top of old code without removing obsolete layers, creating a "lava layer" stack of different technologies and paradigms:

```
┌─────────────────────────────────────┐
│  Vue.js 3 components (2025)         │  ← Current team
├─────────────────────────────────────┤
│  Angular 1.x controllers (2019)     │  ← Previous team
├─────────────────────────────────────┤
│  jQuery widgets (2015)              │  ← Team before that
├─────────────────────────────────────┤
│  Legacy server-rendered HTML (2011) │  ← Original team
└─────────────────────────────────────┘
```

**Symptoms**: Multiple JavaScript frameworks coexist; new developers cannot understand which pattern to follow; performance degrades because multiple rendering systems run simultaneously.

**Resolution**: Establish a deprecation plan for each old layer with a committed end date. Do not add new features in the old style.

### 12.2 The "If It Ain't Broke" Trap

Resistance to necessary updates because the system currently works. This is most dangerous for:
- Security patches ("we haven't been breached yet")
- Dependency upgrades ("the old version still runs")
- Architecture modernization ("why fix what works?")

The risk accumulates invisibly until a critical vulnerability is exploited or a major version jump makes the upgrade 10× harder than it would have been done incrementally.

### 12.3 Heroic Maintenance

A single engineer who "knows the system" handles all production incidents. This creates:
- Single point of failure (what happens when they quit, get sick, or are on vacation?)
- No incentive to improve the system (the hero's status depends on it remaining complex)
- Burn-out for the hero, resentment from peers

**Resolution**: Runbooks, on-call rotation, blameless post-mortems that produce documentation, and deliberate knowledge sharing sessions.

### 12.4 The Boiled Frog

Technical debt is added so gradually that the team never perceives a sudden change to trigger alarm. Each sprint adds a little more debt, each incident is resolved with a quick fix, until suddenly the team is spending 80% of every sprint on maintenance and cannot explain how they got there.

**Prevention**: Track debt metrics continuously (see Section 8.2). Set thresholds that trigger action: "If technical debt ratio exceeds 10%, we pause feature work until it drops below 7%."

---

## 13. Maintenance Metrics

| Metric | Definition | Healthy Target |
|---|---|---|
| **MTTR** (Mean Time to Repair) | Average time from failure detection to resolution | As low as possible; hours for P1 |
| **Change Failure Rate** | % of deployments that cause production incidents | < 5% (Elite teams: < 1%) |
| **Defect Density** | Bugs per KLOC or per feature | Trending down over time |
| **Technical Debt Ratio** | Estimated debt remediation cost / total development cost | < 5% |
| **Code Churn** | % of code changed within a period | High churn in stable modules = concern |
| **Mean Time Between Failures** (MTBF) | Average time between production failures | Trending upward |

The DORA metrics (Deployment Frequency, Lead Time, MTTR, Change Failure Rate) from the *Accelerate* research are the most validated predictor of organizational performance in software delivery.

---

## 14. Case Study: Modernizing a Monolith

**Scenario**: A 15-year-old e-commerce platform built in PHP 5.3. The monolith handles product catalog, orders, payments, and shipping. It has no tests, no API layer, and relies on a MySQL 5.5 database. The team wants to modernize without stopping product development.

**Chosen Strategy**: Strangler Fig + Phased Migration

**Phase 1: Wrap and Stabilize (6 months)**
- Add integration tests using the UI (Selenium) to characterize existing behavior
- Introduce a routing proxy (nginx) in front of the monolith
- Extract user authentication as a standalone service (least risky component)

**Phase 2: Extract High-Value Services (12 months)**
- New product catalog service (Python/FastAPI) — routes 10%, then 50%, then 100% of catalog traffic
- New order service — run in parallel mode (shadow traffic) for 4 weeks before cutover
- Database: CDC (Change Data Capture) streams MySQL changes to new PostgreSQL databases

**Phase 3: Decommission (6 months)**
- Migrate payments to a dedicated payment service (Stripe-backed)
- Retire the PHP monolith module by module
- Shut down legacy MySQL instance after 30-day clean observation period

**Lessons from this case:**
- Shadow traffic parallel testing discovered 3 calculation discrepancies in the legacy code that nobody knew existed
- The authentication extraction took twice as long as estimated due to undocumented session behavior
- The team celebrated "strangle events" (module retirements) which maintained morale during a long project

---

## Summary

Software maintenance is not an afterthought — it consumes the majority of software lifecycle cost and requires deliberate strategy:

- **Four maintenance types**: Corrective (fix bugs), Adaptive (survive environment changes), Perfective (add features), Preventive (reduce future risk)
- **Lehman's Laws**: Complexity grows and quality declines by default; sustained investment is the only counter
- **Legacy systems**: Contain valuable encoded business logic; modernize incrementally with Strangler Fig rather than big-bang rewrites
- **Refactoring**: The safe, incremental alternative to rewriting — always backed by tests
- **Migration strategies**: Phased or parallel are safer than big bang for most systems
- **Deprecation**: Give consumers ample notice; use compiler/runtime warnings and migration guides
- **Metrics**: MTTR and Change Failure Rate are leading indicators of maintenance health

---

## Practice Exercises

1. **Maintenance Classification**: Classify each of the following tasks as Corrective, Adaptive, Perfective, or Preventive, and justify your classification: (a) fixing a crash that occurs when a user uploads a PNG with EXIF data; (b) updating the SMS library after the vendor retires their v1 API; (c) adding dark mode to the UI; (d) refactoring the 3,000-line `OrderProcessor` class into smaller components; (e) adding rate limiting to the API before a marketing campaign launches.

2. **Lehman's Laws Analysis**: Choose a well-known open-source project (e.g., Linux kernel, Python interpreter, Firefox). Research its commit history and release notes. Identify two specific examples where Lehman's Law I (Continuing Change) and Law II (Increasing Complexity) are visible. How has the project counteracted complexity growth?

3. **Strangler Fig Design**: You maintain a monolithic Java application that handles HR functions: payroll, leave management, performance reviews, and onboarding. You want to migrate to microservices. Design a Strangler Fig migration plan: identify which service to extract first and why, describe the routing proxy strategy, and define the criteria for declaring a "strangle event" complete.

4. **Refactoring Practice**: Take the following code and apply at least three named refactoring patterns. Preserve the observable behavior. Document which pattern you applied and why.
   ```python
   def x(d, t, u, s):
       r = 0
       for i in d:
           if i['type'] == t and i['user'] == u and i['status'] == s:
               r += i['amount']
       if r > 1000:
           r = r * 0.95
       return r
   ```

5. **Deprecation Policy Design**: You are the platform team at a company with 50 internal service consumers of your authentication API. You need to deprecate the `GET /auth/token?user=X&pass=Y` endpoint (which sends credentials in query parameters, a security violation) and replace it with `POST /auth/token` with credentials in the request body. Design a complete deprecation plan: timeline, communication strategy, monitoring approach, and hard cutover criteria.

---

## Further Reading

- **Books**:
  - Feathers, M. (2004). *Working Effectively with Legacy Code*. Prentice Hall. (The definitive guide to adding tests to untested code)
  - Fowler, M. (2018). *Refactoring: Improving the Design of Existing Code*, 2nd Ed. Addison-Wesley.
  - Parnas, D. (1994). "Software Aging." *Proceedings of ICSE 1994*. (Original paper on the concept)

- **Articles and Papers**:
  - Fowler, M. "Strangler Fig Application." [martinfowler.com](https://martinfowler.com/bliki/StranglerFigApplication.html)
  - Spolsky, J. "Things You Should Never Do, Part I." [Joel on Software](https://www.joelonsoftware.com/2000/04/06/things-you-should-never-do-part-i/)
  - Forsgren, N., Humble, J. & Kim, G. (2018). *Accelerate*. IT Revolution. (DORA metrics research)

- **Standards**:
  - ISO/IEC 14764:2006 — Software Engineering: Software Life Cycle Processes — Maintenance
  - ISO/IEC 25010 — Systems and Software Quality Models (replaces ISO 9126)

---

**Previous**: [Project Management](./10_Project_Management.md) | **Next**: [Process Improvement](./12_Process_Improvement.md)
