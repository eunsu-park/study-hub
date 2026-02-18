# Lesson 07: Software Quality Assurance

**Previous**: [06. Estimation](./06_Estimation.md) | **Next**: [08. Verification and Validation](./08_Verification_and_Validation.md)

---

Software quality does not emerge accidentally. It is the result of deliberate processes, standards, measurements, and culture applied throughout the software development lifecycle. This lesson explores the discipline of Software Quality Assurance (SQA) — what it means to define, measure, and systematically improve the quality of software products and processes.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Basic understanding of software development lifecycle (Lesson 02)
- Familiarity with software testing concepts
- Basic programming knowledge

**Learning Objectives**:
- Define software quality using IEEE and ISO 25010 frameworks
- Distinguish between quality assurance, quality control, and testing
- Apply the cost of quality model to software projects
- Compute and interpret common software metrics (cyclomatic complexity, cohesion, coupling)
- Use static analysis tools to measure and improve code quality
- Identify and manage technical debt
- Conduct and participate in effective code reviews

---

## Table of Contents

1. [Defining Software Quality](#1-defining-software-quality)
2. [Quality Attributes: ISO/IEC 25010](#2-quality-attributes-isoiec-25010)
3. [QA vs QC vs Testing](#3-qa-vs-qc-vs-testing)
4. [Cost of Quality](#4-cost-of-quality)
5. [The QA Process](#5-the-qa-process)
6. [Software Metrics](#6-software-metrics)
7. [Static Analysis and Code Quality Tools](#7-static-analysis-and-code-quality-tools)
8. [Quality Standards](#8-quality-standards)
9. [Technical Debt](#9-technical-debt)
10. [Code Reviews](#10-code-reviews)
11. [Summary](#11-summary)
12. [Practice Exercises](#12-practice-exercises)
13. [Further Reading](#13-further-reading)

---

## 1. Defining Software Quality

Quality is one of the most contested terms in software engineering. Multiple definitions coexist, each emphasizing a different perspective.

**IEEE Definition (IEEE Std 730)**:
> "The degree to which a system, component, or process meets specified requirements."

This is a *conformance-based* view: quality means fulfilling what was asked.

**Crosby's Definition**:
> "Quality is conformance to requirements."

**Juran's Definition**:
> "Fitness for use."

This shifts focus from documents to user needs — a product can conform to requirements yet still be low quality if the requirements were wrong.

**ISO/IEC 25010 (SQuaRE)**:
Provides a multi-dimensional model of product quality and quality in use (covered in detail in Section 2).

### The Quality Dilemma

These definitions create practical tension:

```
Perspective         Question                  Risk if ignored
──────────────────────────────────────────────────────────────
Conformance         Did we build it right?    Technical failure
Fitness for use     Did we build the right    User rejection
                    thing?
Value               Is it worth building?     Business failure
Excellence          Is it the best it can     Competitive loss
                    be?
```

In practice, a complete quality program addresses all four. Lessons 08 (V&V) addresses the first two in depth; this lesson focuses on the processes and measurements that support all four.

---

## 2. Quality Attributes: ISO/IEC 25010

ISO/IEC 25010 (part of the SQuaRE — System and Software Quality Requirements and Evaluation — series) defines two quality models:

- **Product quality model**: intrinsic characteristics of the software artifact
- **Quality in use model**: outcomes when the system is used in a specific context

### 2.1 Product Quality Model

| Quality Characteristic | Sub-characteristics | What it means |
|------------------------|---------------------|---------------|
| **Functional Suitability** | Completeness, Correctness, Appropriateness | Does the software do what it should? |
| **Reliability** | Maturity, Availability, Fault Tolerance, Recoverability | How well does it perform under expected and unexpected conditions? |
| **Performance Efficiency** | Time Behaviour, Resource Utilisation, Capacity | Is it fast and resource-efficient enough? |
| **Usability** | Appropriateness Recognizability, Learnability, Operability, User Error Protection, User Interface Aesthetics, Accessibility | Can users achieve their goals easily? |
| **Security** | Confidentiality, Integrity, Non-repudiation, Accountability, Authenticity | Does it protect data and resist attacks? |
| **Compatibility** | Co-existence, Interoperability | Does it work alongside other systems? |
| **Maintainability** | Modularity, Reusability, Analysability, Modifiability, Testability | Can it be changed without undue effort? |
| **Portability** | Adaptability, Installability, Replaceability | Can it be moved to a new environment? |

### 2.2 Quality in Use Model

| Characteristic | Description |
|----------------|-------------|
| **Effectiveness** | Can users achieve their goals completely and accurately? |
| **Efficiency** | Do they achieve goals with appropriate resource expenditure? |
| **Satisfaction** | Are users' needs and expectations met? |
| **Freedom from Risk** | Does it mitigate risks to economy, safety, or environment? |
| **Context Coverage** | Does it work across the target range of contexts? |

### 2.3 Prioritizing Quality Attributes

Not all attributes matter equally for every system. A pacemaker firmware prioritizes reliability and safety above all. A marketing landing page prioritizes usability and performance. A banking API prioritizes security and reliability.

Quality attribute prioritization is captured in a **Quality Attribute Utility Tree** (common in architecture practices):

```
Root: System Quality
  ├── Performance
  │     ├── Response time < 200ms under 1000 concurrent users  [HIGH, HIGH]
  │     └── Throughput > 10,000 transactions/min               [HIGH, MEDIUM]
  ├── Security
  │     ├── No SQL injection vulnerabilities                    [HIGH, HIGH]
  │     └── Session tokens expire after 30 minutes             [MEDIUM, LOW]
  └── Maintainability
        └── New feature added in < 2 developer days            [MEDIUM, MEDIUM]
```

The two annotations `[importance, difficulty]` help teams prioritize architectural decisions.

---

## 3. QA vs QC vs Testing

These three terms are often confused. Understanding the distinction is essential for building a coherent quality program.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Quality Assurance (QA)                                             │
│  Process-oriented. Prevents defects from entering the product.      │
│  Examples: coding standards, process audits, training, reviews      │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Quality Control (QC)                                         │  │
│  │  Product-oriented. Identifies defects in the product.        │  │
│  │  Examples: reviews, inspections, walkthroughs                │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Testing                                                │  │  │
│  │  │  Execution-oriented. Finds failures by running code.   │  │  │
│  │  │  Examples: unit tests, integration tests, UAT          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

| Dimension | QA | QC | Testing |
|-----------|----|----|---------|
| Focus | Process | Product | Execution |
| Goal | Prevent defects | Find defects | Find failures |
| When | Throughout SDLC | At checkpoints | After code exists |
| Output | Standards, process docs | Defect reports, review findings | Test results, bug reports |
| Scope | All activities | Artifacts and deliverables | Running software |

**Key insight**: QA is a *managerial function*. A QA team does not just run tests — it defines and enforces the processes that make quality possible. Testing is necessary but not sufficient for quality.

---

## 4. Cost of Quality

Philip Crosby's **Cost of Quality (CoQ)** model argues that quality-related costs fall into four categories. The model reveals a counterintuitive truth: *investing in prevention reduces total cost*.

### 4.1 The Four CoQ Categories

```
Cost of Quality
├── Cost of Conformance (money spent preventing poor quality)
│   ├── Prevention Costs
│   │   ├── Training
│   │   ├── Process documentation
│   │   ├── Coding standards
│   │   ├── Static analysis tools
│   │   └── Architecture reviews
│   └── Appraisal Costs
│       ├── Code reviews
│       ├── Test execution
│       ├── Test infrastructure
│       └── QA audits
│
└── Cost of Non-Conformance (money spent dealing with poor quality)
    ├── Internal Failure Costs (found before release)
    │   ├── Bug fixing
    │   ├── Rework
    │   ├── Regression testing after fixes
    │   └── Delayed releases
    └── External Failure Costs (found by customers after release)
        ├── Customer support
        ├── Patches and hotfixes
        ├── Legal liability
        ├── Reputation damage
        └── Customer churn
```

### 4.2 The 1-10-100 Rule

The cost of fixing a defect escalates dramatically with how late it is found:

| Phase Found | Relative Cost |
|-------------|---------------|
| Requirements | 1x |
| Design | 5x |
| Coding | 10x |
| Unit testing | 15x |
| Integration testing | 25x |
| System testing | 50x |
| Production | 100x+ |

This is why early investment in QA (requirements reviews, design reviews, static analysis) yields high returns even though it feels expensive upfront.

### 4.3 Optimal Quality Investment

The classic economic model shows that total CoQ has a minimum at some non-zero defect rate:

```
Cost
 ▲
 │     Total Cost
 │       ╲     ╱
 │        ╲   ╱
 │  Non-   ╲ ╱ Conformance
 │  conform  X
 │  cost    ╱ ╲
 │         ╱   ╲ Prevention/
 │        ╱     ╲ Appraisal cost
 │───────────────────────────────▶ Defect Rate
         ↑
      Optimum
```

However, modern Agile and DevOps practices shift this curve: automation makes prevention cheaper, pushing the optimum toward near-zero defects.

---

## 5. The QA Process

A formal SQA program typically includes these activities:

### 5.1 SQA Planning

The **Software Quality Assurance Plan (SQAP)** (IEEE Std 730) documents:
- Quality standards and metrics to be applied
- Reviews, audits, and testing to be performed
- Who is responsible for each QA activity
- Procedures for reporting and tracking defects
- Tools to be used

### 5.2 Standards and Procedures

SQA establishes:
- **Coding standards**: naming conventions, formatting, documentation requirements
- **Process standards**: how requirements are written, how designs are reviewed
- **Documentation standards**: what documents are required and their formats

### 5.3 Reviews and Audits

| Activity | Purpose | Participants |
|----------|---------|--------------|
| **Requirements review** | Validate completeness, consistency, testability | Authors, analysts, customer |
| **Design review** | Evaluate architecture and design decisions | Architects, developers, QA |
| **Code review** | Find defects, enforce standards | Developers, peers |
| **Test plan review** | Validate test coverage and approach | QA, developers, PM |
| **Process audit** | Verify team is following defined processes | QA lead, management |
| **Product audit** | Verify deliverable meets standards | QA, customer |

### 5.4 Metrics Collection and Analysis

SQA collects metrics continuously to detect process problems early. If defect density spikes after a refactoring, that signal triggers investigation before the next release.

---

## 6. Software Metrics

Software metrics quantify aspects of the software product or process, enabling objective assessment and trend tracking.

### 6.1 Product Metrics

#### Cyclomatic Complexity (McCabe, 1976)

Measures the number of linearly independent paths through a program's source code.

```
CC = E - N + 2P

Where:
  E = number of edges in the control flow graph
  N = number of nodes
  P = number of connected components (usually 1)
```

For a single function, this simplifies to: **CC = number of decision points + 1**

```python
def classify_triangle(a, b, c):          # CC starts at 1
    if a <= 0 or b <= 0 or c <= 0:       # +1 → 2
        return "Invalid"
    if a + b <= c or b + c <= a or a + c <= b:  # +1, +1, +1 → 5
        return "Not a triangle"
    if a == b == c:                       # +1 → 6
        return "Equilateral"
    elif a == b or b == c or a == c:      # +1, +1, +1 → 9
        return "Isosceles"
    else:
        return "Scalene"
# CC = 9
```

| CC Value | Risk | Recommended Action |
|----------|------|--------------------|
| 1–10 | Low | Acceptable |
| 11–20 | Moderate | Consider refactoring |
| 21–50 | High | Refactor |
| > 50 | Very High | Must refactor; untestable |

#### Coupling and Cohesion

These two metrics are inverses of each other in quality terms:

**Cohesion** — how strongly related the responsibilities within a single module are.

```
High Cohesion (Good)           Low Cohesion (Bad)
─────────────────────          ─────────────────────
UserAuthenticator               UtilityHelper
  + login()                       + login()
  + logout()                      + formatDate()
  + resetPassword()               + sendEmail()
  + validateToken()               + parseCSV()
                                  + calculateTax()
```

**Coupling** — how much one module depends on another.

```
Low Coupling (Good)            High Coupling (Bad)
──────────────────────         ──────────────────────
OrderService → IPayment        OrderService → PaypalPayment
(depends on interface)         (depends on concrete class,
                                internal state, and DB schema)
```

| Cohesion Type | Description | Quality |
|---------------|-------------|---------|
| Functional | All elements work toward one well-defined task | Best |
| Sequential | Output of one element feeds the next | Good |
| Communicational | Elements operate on the same data | Moderate |
| Procedural | Elements follow a fixed execution order | Moderate |
| Temporal | Elements are grouped by when they execute | Poor |
| Logical | Elements selected by a control flag | Poor |
| Coincidental | No meaningful relationship | Worst |

#### Halstead Metrics

Halstead (1977) measures software based on token counts:

```
n1 = number of distinct operators
n2 = number of distinct operands
N1 = total occurrences of operators
N2 = total occurrences of operands

Vocabulary:  n  = n1 + n2
Length:      N  = N1 + N2
Volume:      V  = N × log2(n)
Difficulty:  D  = (n1/2) × (N2/n2)
Effort:      E  = D × V
```

Halstead volume correlates with the effort required to understand and modify a module.

### 6.2 Process Metrics

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **Defect Density** | Defects / KLOC | Bug rate per 1000 lines |
| **Defect Removal Efficiency (DRE)** | Defects found before release / (before + after) × 100% | % of defects caught before shipping |
| **Mean Time Between Failures (MTBF)** | Total uptime / Number of failures | Reliability of deployed system |
| **Mean Time to Repair (MTTR)** | Total repair time / Number of failures | Responsiveness to failures |
| **Test Coverage** | Exercised code / Total code × 100% | How much code tests exercise |
| **Escaped Defect Rate** | Production bugs / Total bugs × 100% | Defects missed by all QA activities |

Industry benchmark for DRE: world-class organizations achieve 95–99%.

---

## 7. Static Analysis and Code Quality Tools

Static analysis examines source code without executing it, finding defects, style violations, security vulnerabilities, and complexity issues.

### 7.1 Categories of Static Analysis

```
Static Analysis
├── Style and Formatting
│   └── Does the code follow agreed conventions?
│   └── Tools: Prettier, Black, gofmt, clang-format
│
├── Linting
│   └── Are there common programming errors or anti-patterns?
│   └── Tools: ESLint, Pylint, Flake8, RuboCop, Checkstyle
│
├── Complexity Analysis
│   └── Is any function too complex to understand or test?
│   └── Tools: Lizard, Radon, SonarQube
│
├── Security Scanning (SAST)
│   └── Are there known vulnerability patterns (SQL injection, XSS, etc.)?
│   └── Tools: Bandit (Python), Semgrep, CodeQL, Checkmarx
│
└── Dependency Scanning
    └── Do any dependencies have known CVEs?
    └── Tools: Dependabot, Snyk, OWASP Dependency-Check
```

### 7.2 SonarQube: An Integrated Platform

SonarQube consolidates many static analysis concerns into one dashboard. Key concepts:

| SonarQube Concept | Meaning |
|-------------------|---------|
| **Bug** | Code that is likely wrong and will cause a runtime failure |
| **Vulnerability** | Code that is susceptible to an attack |
| **Code Smell** | Code that is maintainable but needlessly complex or confusing |
| **Security Hotspot** | Code that needs human review for security context |
| **Technical Debt** | Estimated remediation time for all issues |
| **Quality Gate** | Pass/fail threshold that CI/CD enforces |

A typical Quality Gate configuration:

```yaml
# sonar-project.properties
sonar.qualitygate.wait=true

# Quality Gate conditions:
# - Coverage on new code >= 80%
# - Duplicated lines on new code < 3%
# - Maintainability rating on new code = A
# - Reliability rating on new code = A
# - Security rating on new code = A
```

### 7.3 Integrating Static Analysis into CI/CD

```yaml
# GitHub Actions example
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run linter
        run: |
          pip install flake8 pylint
          flake8 src/ --max-line-length=100
          pylint src/ --fail-under=8.0

      - name: Run security scan
        run: |
          pip install bandit
          bandit -r src/ -ll  # report medium and high severity only

      - name: Check complexity
        run: |
          pip install radon
          radon cc src/ -a -n C  # fail if average CC > C (10)

      - name: SonarQube Scan
        uses: SonarSource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}
```

---

## 8. Quality Standards

### 8.1 ISO 9001:2015

ISO 9001 is a general quality management standard applicable to any organization. For software companies, it requires:

- Documented quality management system
- Evidence of process compliance
- Corrective and preventive action processes
- Management review of quality metrics
- Customer focus and satisfaction measurement

ISO 9001 certification is often required to do business with government agencies or large enterprises.

### 8.2 ISO/IEC 25010 (SQuaRE)

Already covered in Section 2. This standard is the foundation for defining quality requirements and evaluation criteria for software products.

### 8.3 CMMI (Capability Maturity Model Integration)

CMMI is a process improvement framework developed by Carnegie Mellon's Software Engineering Institute. It defines five maturity levels:

```
Level 5: Optimizing    ← Continuous process improvement
Level 4: Quantitatively Managed ← Measured and controlled
Level 3: Defined       ← Documented, standardized processes
Level 2: Managed       ← Planned and tracked
Level 1: Initial       ← Ad hoc, chaotic
```

| Level | Characteristics | Typical Organization |
|-------|----------------|----------------------|
| 1 Initial | Success depends on heroics; no repeatable processes | Startup |
| 2 Managed | Basic project management; repeatable per-project | Small company |
| 3 Defined | Organization-wide standard process; tailored per project | Mid-size company |
| 4 Quantitatively Managed | Statistical process control; predictable quality | Mature company |
| 5 Optimizing | Data-driven continuous improvement | World-class org |

Most commercial software organizations operate at Level 2 or 3. Defense and aerospace contractors often require Level 3–5.

---

## 9. Technical Debt

Ward Cunningham coined "technical debt" in 1992 as a metaphor: taking shortcuts now is like borrowing money — you get the benefit today but pay interest (reduced velocity, bugs, brittleness) over time.

### 9.1 Types of Technical Debt

Martin Fowler's Technical Debt Quadrant:

```
                    Reckless            Prudent
                 ┌──────────────────┬──────────────────┐
    Deliberate   │ "We don't have   │ "We must ship    │
                 │ time for design" │ now and deal     │
                 │                  │ with consequences│
                 ├──────────────────┼──────────────────┤
    Inadvertent  │ "What's          │ "Now we know     │
                 │ layering?"       │ how we should    │
                 │                  │ have done it"    │
                 └──────────────────┴──────────────────┘
```

- **Deliberate Reckless**: Dangerous — teams know better but cut corners anyway
- **Deliberate Prudent**: Acceptable — conscious decision with a plan to repay
- **Inadvertent Reckless**: Common — team lacks skills or knowledge
- **Inadvertent Prudent**: Unavoidable — lessons learned during development

### 9.2 Measuring Technical Debt

SonarQube estimates debt as time-to-fix. Other approaches:

```python
# Simple debt estimation model
def estimate_debt_hours(metrics):
    debt = 0

    # Complex functions: 30 min per function above CC threshold
    complex_functions = sum(1 for cc in metrics['cyclomatic']
                           if cc > 10)
    debt += complex_functions * 0.5

    # Low test coverage: 20 min per uncovered function
    uncovered = metrics['total_functions'] * (1 - metrics['coverage'])
    debt += uncovered * 0.33

    # Duplicated code: 1 hour per duplicate block
    debt += metrics['duplicate_blocks'] * 1.0

    # Known code smells from static analysis
    debt += metrics['code_smells'] * 0.25

    return debt  # in hours
```

### 9.3 Managing Technical Debt

Strategies used in practice:

| Strategy | Description | When to use |
|----------|-------------|-------------|
| **Boy Scout Rule** | Leave code cleaner than you found it; fix small things while passing | Always |
| **Debt Sprints** | Dedicate a sprint (or 20% of each sprint) to debt reduction | Regularly |
| **Feature Freeze** | Halt new features; spend a release cycle on quality | When quality is critically degraded |
| **Strangler Fig** | Gradually replace a legacy subsystem with clean code | Large legacy systems |
| **Refactor on Touch** | Refactor a module before adding new features to it | When modules are about to change |

The key discipline: **make debt visible**. Track it in your issue tracker. Give it story points. Include debt reduction in velocity calculations.

---

## 10. Code Reviews

Code reviews are one of the most cost-effective defect prevention techniques available. Studies consistently show that code reviews find 60–90% of defects before tests run.

### 10.1 Types of Code Reviews

| Type | Formality | Effort | Best for |
|------|-----------|--------|----------|
| **Pair Programming** | Informal | Continuous | High-risk code; knowledge transfer |
| **Over-the-Shoulder** | Informal | Low | Quick sanity checks |
| **Tool-Assisted Review** (PR review) | Semi-formal | Medium | Day-to-day development |
| **Walkthrough** | Formal | Medium | Education; broad review |
| **Fagan Inspection** | Formal | High | Safety-critical code |

### 10.2 Pull Request Review Checklist

```
Code Correctness
  □ Does it do what the ticket/story requires?
  □ Are edge cases handled (null, empty, overflow)?
  □ Is error handling appropriate and consistent?
  □ Are there race conditions in concurrent code?

Code Quality
  □ Is the code readable without needing comments?
  □ Are variable/function names descriptive?
  □ Is there duplicated logic that should be extracted?
  □ Is cyclomatic complexity acceptable (< 10 per function)?

Testing
  □ Are there unit tests for the new behavior?
  □ Do tests cover failure cases, not just happy paths?
  □ Is test coverage maintained or improved?

Security
  □ Is user input validated/sanitized?
  □ Are secrets/credentials absent from the code?
  □ Are authorization checks correct?

Performance
  □ Are there obvious N+1 query problems?
  □ Are expensive operations cached appropriately?

Documentation
  □ Are public APIs documented?
  □ Are complex algorithms explained?
  □ Is the CHANGELOG or release notes updated?
```

### 10.3 Making Reviews Effective

**For reviewers**:
- Review in focused sessions of 60–90 minutes maximum (attention degrades after that)
- Focus on logic, design, and correctness — automate style checking
- Ask questions rather than making demands ("What happens if X is null?" not "This is wrong")
- Acknowledge good code, not only problems

**For authors**:
- Keep PRs small (< 400 lines of net new code is a common guideline)
- Write a clear description: what, why, and how
- Self-review before requesting review
- Link to the relevant issue/ticket

**PR size guideline**:
```
Lines of Change    Review Quality Impact
< 50               Thorough review; almost all bugs caught
50–400             Good review; most bugs caught
400–800            Moderate; reviewers skim long sections
> 800              Cursory; many bugs missed; reviewers approve to end pain
```

---

## 11. Summary

Software quality is a multidimensional property defined by standards (ISO 25010), measured with metrics (cyclomatic complexity, defect density, DRE), and systematically managed through QA processes.

Key takeaways:

- **QA is process-focused** (prevents defects); QC is product-focused (detects defects); testing is execution-focused (finds failures).
- **Invest early**: the Cost of Quality model shows that prevention and appraisal are far cheaper than failure costs, especially external failure.
- **Measure what matters**: cyclomatic complexity flags untestable code; coupling and cohesion reveal design problems; DRE measures how effective your QA program is overall.
- **Use tools automatically**: static analysis integrated into CI/CD catches a large class of problems without manual effort.
- **Technical debt is real cost**: make it visible, track it, and allocate capacity to reduce it alongside feature work.
- **Code reviews are high-ROI**: small, focused reviews with clear checklists catch the defects that automated tools miss.

---

## 12. Practice Exercises

**Exercise 1 — Cyclomatic Complexity**

Calculate the cyclomatic complexity of the following Python function. Then refactor it to reduce CC to 4 or below.

```python
def process_order(order):
    if order is None:
        return None
    if order.status == "cancelled":
        return {"error": "Order is cancelled"}
    if order.items:
        for item in order.items:
            if item.quantity < 0:
                raise ValueError(f"Invalid quantity for {item.name}")
            if item.price < 0:
                raise ValueError(f"Invalid price for {item.name}")
            if item.quantity == 0:
                continue
            item.total = item.price * item.quantity
    if order.discount_code:
        if order.discount_code == "SAVE10":
            order.discount = 0.10
        elif order.discount_code == "SAVE20":
            order.discount = 0.20
        else:
            return {"error": "Invalid discount code"}
    order.total = sum(i.total for i in order.items if i.quantity > 0)
    if order.discount:
        order.total *= (1 - order.discount)
    return order
```

**Exercise 2 — Cost of Quality Analysis**

A software team has the following monthly cost data:
- Coding standards enforcement: $2,000
- Code review time: $8,000
- Unit testing: $5,000
- Bug fixing before release: $15,000
- Customer support for production bugs: $25,000
- Emergency hotfix deployments: $10,000

(a) Categorize each cost into Prevention, Appraisal, Internal Failure, or External Failure.
(b) Calculate the total Cost of Quality and the proportion of each category.
(c) If investing an additional $5,000/month in prevention reduces internal failures by 40% and external failures by 25%, should the team make this investment? Show your reasoning.

**Exercise 3 — Quality Gate Design**

You are the QA lead for a team building a fintech API. Design a SonarQube Quality Gate for this system. Specify at least six conditions with thresholds and justify each choice based on the domain.

**Exercise 4 — Technical Debt Backlog**

Review the following code smells found by a static analysis tool on a legacy e-commerce application:
- 47 functions with CC > 15 (average CC = 22)
- Test coverage: 34%
- 280 duplicated code blocks
- 3 potential SQL injection vulnerabilities
- 12 uses of deprecated API methods

(a) Categorize each item using Fowler's Technical Debt Quadrant.
(b) Prioritize the list for a team with two developers who can dedicate 20% of their time to debt reduction.
(c) Write a 3-sprint debt reduction plan.

**Exercise 5 — Code Review**

Review the following Python function as if you were a senior engineer. Identify at least five issues across correctness, quality, security, and performance dimensions. For each issue, explain what could go wrong and propose a fix.

```python
def get_user_data(user_id):
    conn = sqlite3.connect("app.db")
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    result = conn.execute(query).fetchall()
    user = {}
    for row in result:
        user['id'] = row[0]
        user['name'] = row[1]
        user['email'] = row[2]
        user['password'] = row[3]
        user['admin'] = row[4]
    return user
```

---

## 13. Further Reading

- **Books**:
  - *Code Complete* (2nd ed.) — Steve McConnell. Chapters 24–25 cover quality assurance comprehensively.
  - *Clean Code* — Robert C. Martin. Practical guidance on writing maintainable code.
  - *The Art of Software Testing* — Glenford Myers et al.
  - *Working Effectively with Legacy Code* — Michael Feathers. The definitive guide to managing technical debt.

- **Standards**:
  - ISO/IEC 25010:2011 — Systems and software Quality Requirements and Evaluation (SQuaRE)
  - IEEE Std 730-2014 — IEEE Standard for Software Quality Assurance Processes
  - CMMI Institute — https://cmmiinstitute.com/

- **Tools**:
  - SonarQube Community Edition — https://www.sonarqube.org/
  - Radon (Python complexity) — https://radon.readthedocs.io/
  - Lizard (multi-language complexity) — https://github.com/terryyin/lizard
  - Semgrep (pattern-based static analysis) — https://semgrep.dev/

- **Papers**:
  - McCabe, T. J. (1976). "A Complexity Measure." *IEEE Transactions on Software Engineering*.
  - Fagan, M. E. (1976). "Design and Code Inspections to Reduce Errors in Program Development." *IBM Systems Journal*.
  - Cunningham, W. (1992). "The WyCash Portfolio Management System." *OOPSLA*.

---

**Previous**: [06. Estimation](./06_Estimation.md) | **Next**: [08. Verification and Validation](./08_Verification_and_Validation.md)
