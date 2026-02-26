# Lesson 08: Verification and Validation

**Previous**: [07. Software Quality Assurance](./07_Software_Quality_Assurance.md) | **Next**: [09. Configuration Management](./09_Configuration_Management.md)

---

"Are we building the product right?" and "Are we building the right product?" These two questions, posed by Barry Boehm in 1979, capture the essence of Verification and Validation (V&V). Together they form the quality backbone of software engineering — a systematic, disciplined approach to ensuring that software is both technically correct and genuinely useful. This lesson surveys the full spectrum of V&V techniques, from unit tests to formal proofs, and equips you to design a comprehensive test strategy for real-world projects.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Software Quality Assurance fundamentals (Lesson 07)
- Basic programming and unit testing experience
- Familiarity with software development lifecycle (Lesson 02)

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish verification from validation and articulate why both are necessary
2. Design test cases using black-box techniques (equivalence partitioning, boundary value analysis, decision tables, state transition)
3. Select appropriate coverage criteria for white-box testing
4. Plan a test strategy that spans unit, integration, system, and acceptance testing
5. Understand the bug lifecycle and write effective defect reports
6. Evaluate the role of formal methods in high-assurance software
7. Build a regression test suite and integrate it into automated pipelines

---

## Table of Contents

1. [Verification vs Validation](#1-verification-vs-validation)
2. [Testing Levels](#2-testing-levels)
3. [Testing Types](#3-testing-types)
4. [Black-Box Testing Techniques](#4-black-box-testing-techniques)
5. [White-Box Testing Techniques](#5-white-box-testing-techniques)
6. [Test Planning and Documentation](#6-test-planning-and-documentation)
7. [Test-Driven Development](#7-test-driven-development)
8. [Regression Testing and Test Automation](#8-regression-testing-and-test-automation)
9. [Reviews and Inspections](#9-reviews-and-inspections)
10. [Formal Verification](#10-formal-verification)
11. [The Bug Lifecycle](#11-the-bug-lifecycle)
12. [Summary](#12-summary)
13. [Practice Exercises](#13-practice-exercises)
14. [Further Reading](#14-further-reading)

---

## 1. Verification vs Validation

### 1.1 The Core Distinction

| | Verification | Validation |
|--|--------------|------------|
| **Question** | Are we building the product right? | Are we building the right product? |
| **Reference** | Specification, design documents | User needs, business goals |
| **Activities** | Reviews, inspections, static analysis, testing against spec | User acceptance testing, beta testing, prototyping |
| **Phase** | Throughout development | Primarily at milestones and delivery |
| **Finds** | Does the system conform to its specification? | Does the specification reflect what users actually need? |

Both are essential. A system that perfectly implements its specification but fails to meet user needs is a validated failure — the specification was wrong. A system that meets user needs but is built without verification is a pile of technical debt and latent defects.

### 1.2 The V-Model

The V-Model maps development phases (left leg of the V) to corresponding test phases (right leg), making the verification/validation relationship explicit:

```
Requirements Analysis ──────────────────────── Acceptance Testing
        │                                              │
   System Design ──────────────────── System Testing  │
        │                                    │         │
  Architecture Design ──── Integration Testing        │
        │                          │                   │
    Detailed Design ─── Unit Testing                   │
        │                    │                         │
      Coding ────────────────┘                         │
        │                                              │
        └──────── Development ──────── Testing ────────┘
```

Each test phase validates the output of the corresponding development phase. Acceptance testing validates requirements; system testing validates system design; integration testing validates architecture; unit testing validates detailed design.

### 1.3 V&V Independence

The IEEE standard on V&V (IEEE Std 1012) distinguishes between:

- **Independent V&V (IV&V)**: performed by a group with no stake in the development outcome — often a separate organization. Required for safety-critical systems (medical devices, aerospace, nuclear).
- **Non-independent V&V**: performed by the development team or organization. Suitable for most commercial software.

Studies show that IV&V catches a significantly higher percentage of defects because reviewers have no blind spots from writing the code.

---

## 2. Testing Levels

Testing is organized into four hierarchical levels, each with a distinct scope and purpose.

### 2.1 Unit Testing

**Scope**: A single function, method, or class — the smallest testable unit.

**Who**: The developer who wrote the code (or pair partner in TDD).

**Goal**: Verify that individual units behave correctly in isolation.

**Characteristics**:
- Fast (milliseconds per test)
- No external dependencies (databases, network, file system are mocked)
- Highly focused; a failing test points to a specific function

```python
# pytest example: testing a utility function in isolation
from decimal import Decimal
import pytest
from pricing import apply_discount

class TestApplyDiscount:
    def test_percentage_discount(self):
        price = Decimal("100.00")
        result = apply_discount(price, discount_pct=10)
        assert result == Decimal("90.00")

    def test_zero_discount(self):
        price = Decimal("50.00")
        result = apply_discount(price, discount_pct=0)
        assert result == Decimal("50.00")

    def test_hundred_percent_discount(self):
        result = apply_discount(Decimal("75.00"), discount_pct=100)
        assert result == Decimal("0.00")

    def test_negative_discount_raises(self):
        with pytest.raises(ValueError, match="discount must be non-negative"):
            apply_discount(Decimal("100.00"), discount_pct=-5)

    def test_discount_above_100_raises(self):
        with pytest.raises(ValueError, match="discount cannot exceed 100"):
            apply_discount(Decimal("100.00"), discount_pct=110)
```

### 2.2 Integration Testing

**Scope**: Interactions between units, modules, or subsystems.

**Goal**: Verify that components work correctly together — catch interface mismatches, contract violations, and integration-level bugs.

**Approaches**:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Big Bang** | Integrate everything at once, then test | Simple to set up | Hard to locate failures |
| **Top-Down** | Integrate from top-level modules downward; stub lower modules | Tests high-level logic early | Stubs can hide lower-level bugs |
| **Bottom-Up** | Integrate from lower modules upward; use test drivers | Tests real lower-level behavior early | High-level logic tested late |
| **Sandwich (Hybrid)** | Top-down and bottom-up simultaneously | Balances both | More complex planning |
| **Incremental** | Integrate one component at a time | Failures easy to locate | More effort planning integration order |

### 2.3 System Testing

**Scope**: The complete, integrated system as a whole.

**Goal**: Validate that the entire system meets its requirements — functional and non-functional.

System testing includes both functional tests (does it do the right thing?) and non-functional tests (is it fast enough? secure enough? available enough?). It is typically performed by a dedicated QA team, not the developers.

### 2.4 Acceptance Testing

**Scope**: The complete system from the user's perspective.

**Goal**: Validate that the system meets user needs and business requirements.

| Type | Who performs it | Purpose |
|------|-----------------|---------|
| **User Acceptance Testing (UAT)** | End users or their representatives | Confirm the system works for real tasks |
| **Alpha Testing** | Internal users (company employees outside dev team) | Find bugs before external release |
| **Beta Testing** | Selected external users | Find bugs under real-world conditions |
| **Contract Acceptance Testing** | Customer, per contract terms | Verify contractual obligations |
| **Regulation Acceptance Testing** | Regulatory authority | Verify compliance with regulations |

---

## 3. Testing Types

Testing types cut across levels and classify tests by what property they verify.

### 3.1 Functional Testing

Verifies that the system does what it should do — checks features against requirements.

- **Smoke testing**: a quick set of tests that verify the system can start and perform basic operations. Run after every build to decide if deeper testing is warranted.
- **Sanity testing**: a subset of regression testing to verify a specific fix works.
- **Feature testing**: systematic testing of all features described in requirements.

### 3.2 Non-Functional Testing

| Type | What it measures | Key question |
|------|-----------------|--------------|
| **Performance testing** | Speed, throughput, resource usage | Can it handle the expected load? |
| **Load testing** | Behavior under expected peak load | Does it degrade gracefully at peak? |
| **Stress testing** | Behavior under extreme or unexpected load | Where does it break? How does it fail? |
| **Soak/endurance testing** | Behavior over extended time at normal load | Are there memory leaks or slow degradation? |
| **Scalability testing** | How capacity grows with added resources | Does performance scale linearly with servers? |
| **Security testing** | Resistance to attacks | Is it vulnerable to OWASP Top 10? |
| **Usability testing** | Ease of use | Can users accomplish tasks without confusion? |
| **Accessibility testing** | Compliance with WCAG | Can users with disabilities use the system? |
| **Compatibility testing** | Operation across environments | Does it work on Chrome, Firefox, Safari, iOS, Android? |
| **Recovery testing** | Behavior after failure | Does it recover correctly from a crash? |

### 3.3 Performance Testing Example

```python
# locust: load testing tool for web applications
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)  # simulate 1-3 seconds think time

    @task(3)  # weight: this task runs 3x more often than weight-1 tasks
    def browse_products(self):
        self.client.get("/api/products?page=1&limit=20")

    @task(1)
    def view_product_detail(self):
        product_id = 42
        self.client.get(f"/api/products/{product_id}")

    @task(1)
    def add_to_cart(self):
        self.client.post("/api/cart/items", json={
            "product_id": 42,
            "quantity": 1
        })

# Run: locust -f locustfile.py --headless -u 1000 -r 100 --host http://localhost:8000
# -u 1000: 1000 concurrent users
# -r 100: ramp up at 100 users/second
```

---

## 4. Black-Box Testing Techniques

Black-box techniques derive test cases from the specification without knowledge of the internal implementation. They answer: "What inputs should I try?"

### 4.1 Equivalence Partitioning

Divide the input domain into classes where the system should behave the same way for all members. Test one value from each class.

**Example**: A function that accepts age (integer) to determine a pricing tier.
- Valid ages: 0–12 (child), 13–17 (teen), 18–64 (adult), 65+ (senior)
- Invalid: negative, non-integer, null

```
Partition              Representative Value    Expected Result
───────────────────────────────────────────────────────────────
Valid: child (0–12)    8                       "child" tier
Valid: teen (13–17)    15                      "teen" tier
Valid: adult (18–64)   30                      "adult" tier
Valid: senior (65+)    70                      "senior" tier
Invalid: negative      -1                      ValueError
Invalid: null          None                    TypeError
```

Equivalence partitioning reduces the test space from infinite to manageable while maintaining coverage of distinct behaviors.

### 4.2 Boundary Value Analysis

Bugs cluster at the boundaries of equivalence classes. Test values at, just below, and just above each boundary.

Using the age example:
```
Boundary  Values to test
────────────────────────────────────────
0 (lower bound of child)     -1, 0, 1
12/13 (child/teen boundary)  12, 13
17/18 (teen/adult boundary)  17, 18
64/65 (adult/senior bound.)  64, 65
Max (e.g., 150)              149, 150, 151
```

**Why boundaries matter**: Off-by-one errors (`<` vs `<=`, `>` vs `>=`) are among the most common programming mistakes, and they only manifest at boundaries.

### 4.3 Decision Table Testing

Decision tables systematically enumerate combinations of conditions and their corresponding actions. They prevent "combination blindness" — missing the interaction between two conditions.

**Example**: A loan approval system with three conditions.

| | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|--|----|----|----|----|----|----|----|----|
| Credit score > 700 | T | T | T | T | F | F | F | F |
| Income > $50k | T | T | F | F | T | T | F | F |
| Debt ratio < 40% | T | F | T | F | T | F | T | F |
| **Action** | Approve | Approve with review | Approve with review | Reject | Approve with review | Reject | Reject | Reject |

Each column (rule) becomes a test case. Collapsed rules (same action for multiple conditions) can be merged, but each distinct action needs at least one test.

### 4.4 State Transition Testing

For systems with distinct states, derive test cases from the state machine. Test valid transitions, invalid transitions, and boundary states.

**Example**: An order state machine.

```
        ┌──────────────────────────────────────────────┐
        │                                              │
      [New] ──── pay ────▶ [Paid] ──── ship ───▶ [Shipped]
        │                    │                        │
     cancel               cancel                  deliver
        │                    │                        │
        ▼                    ▼                        ▼
   [Cancelled]         [Cancelled]            [Delivered]
                                                      │
                                                    return
                                                      │
                                                      ▼
                                               [Returned]
```

Test cases to cover:
1. New → pay → Paid (valid)
2. New → cancel → Cancelled (valid)
3. Paid → ship → Shipped (valid)
4. Paid → cancel → Cancelled (valid)
5. Shipped → deliver → Delivered (valid)
6. Delivered → return → Returned (valid)
7. New → ship (invalid transition — should be rejected)
8. Cancelled → pay (invalid transition from terminal state)

---

## 5. White-Box Testing Techniques

White-box (structural) techniques derive test cases from the source code's internal structure. They measure *coverage* — how much of the code is exercised.

### 5.1 Statement Coverage

Every executable statement is executed at least once.

```python
def classify(x):
    result = "unknown"          # Statement 1
    if x > 0:                   # Statement 2
        result = "positive"     # Statement 3
    elif x < 0:                 # Statement 4
        result = "negative"     # Statement 5
    else:
        result = "zero"         # Statement 6
    return result               # Statement 7
```

For 100% statement coverage, need at least 3 test cases: x=1, x=-1, x=0.

**Weakness**: Statement coverage can be achieved without testing false branches.

### 5.2 Branch Coverage (Decision Coverage)

Every branch from every decision point is taken at least once (both the true and false outcome of every `if`, `while`, `for`).

Branch coverage subsumes statement coverage: 100% branch coverage implies 100% statement coverage, but not vice versa.

```python
def validate_age(age):
    if age is None:             # Branch: True (age is None), False (age is not None)
        return False
    if age < 0 or age > 150:   # Branch: True, False; compound condition
        return False
    return True
```

For 100% branch coverage:
- `validate_age(None)` — True branch of first if
- `validate_age(25)` — False branch of first if, False branch of second if
- `validate_age(-1)` — True branch of second if

### 5.3 Condition Coverage

Every boolean sub-condition (predicate) evaluates to both True and False independently.

For compound conditions like `age < 0 or age > 150`:
- Need `age < 0` to be True and False
- Need `age > 150` to be True and False

### 5.4 Path Coverage

Every distinct path through the code is executed. For a function with `n` independent decision points, there are up to `2^n` paths.

Path coverage is the strongest criterion but is usually impractical for real functions (exponential explosion). It is used selectively for safety-critical modules.

### 5.5 Coverage Summary

```
Coverage Criterion    Strength    Practical Use
──────────────────────────────────────────────────────────────
Statement             Weakest     Minimum acceptable (80–90%)
Branch                Moderate    Standard for most projects
Condition             Stronger    Security-critical code
MC/DC*                Strong      DO-178C (avionics), safety systems
Path                  Strongest   Impractical except for small units
```

*MC/DC = Modified Condition/Decision Coverage, required by FAA for avionics software.

### 5.6 Measuring Coverage in Practice

```bash
# Python: pytest + coverage
pip install pytest pytest-cov

pytest --cov=src --cov-report=html --cov-fail-under=80

# Output:
# Name                 Stmts   Miss  Cover
# ────────────────────────────────────────
# src/pricing.py          45      3    93%
# src/inventory.py        72     18    75%
# src/checkout.py         98     12    88%
# ────────────────────────────────────────
# TOTAL                  215     33    85%
```

```bash
# JavaScript: Jest
jest --coverage --coverageThreshold='{"global":{"branches":80,"lines":80}}'
```

---

## 6. Test Planning and Documentation

### 6.1 The Test Plan

A test plan (IEEE Std 829) documents the scope, approach, resources, and schedule for testing. Key sections:

| Section | Content |
|---------|---------|
| **Test Scope** | What features/components are in scope and out of scope |
| **Test Approach** | Testing levels, types, techniques to be used |
| **Entry/Exit Criteria** | When testing begins and when it is complete |
| **Test Environment** | Hardware, OS, browsers, network configuration |
| **Resources** | Who performs which tests; tools required |
| **Schedule** | Timeline for each testing phase |
| **Risk and Contingency** | Risks to the test effort and mitigation plans |
| **Deliverables** | Test cases, test data, defect reports, test summary report |

**Entry criteria** example:
- All code for the sprint is merged and passing CI
- Unit test coverage ≥ 80%
- No open critical/high defects from the previous cycle

**Exit criteria** example:
- All planned test cases executed
- No open critical or high-severity defects
- Defect removal efficiency ≥ 90%
- Performance test results within 10% of targets

### 6.2 Writing Test Cases

A good test case is atomic, independent, and reproducible.

```
Test Case ID:    TC-CHECKOUT-042
Title:           Cart checkout fails gracefully when payment service is down
Preconditions:   - User is logged in
                 - Cart has 2 items totaling $59.98
                 - Payment service mock is configured to return 503
Feature:         Checkout / Payment
Test Data:       User: testuser@example.com, Cart: [item_id:1, item_id:7]
Steps:
  1. Navigate to /cart
  2. Click "Proceed to Checkout"
  3. Enter valid credit card: 4111 1111 1111 1111
  4. Click "Place Order"
Expected Result: System displays "Payment service temporarily unavailable.
                 Your cart has been saved. Please try again in a few minutes."
                 No order is created in the database.
                 Cart contents are preserved.
Actual Result:   (filled in during test execution)
Status:          Pass / Fail / Blocked
Severity:        High
Priority:        High
Author:          J. Smith
Date:            2024-03-15
```

### 6.3 Test Data Management

Good test data is:
- **Representative**: covers all equivalence partitions
- **Reproducible**: the same data produces the same result
- **Isolated**: tests do not share mutable state
- **Anonymized**: uses synthetic data, not real production data

```python
# Using factories for reproducible test data (Factory Boy library)
import factory
from factory.django import DjangoModelFactory
from myapp.models import User, Order

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user_{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    is_active = True

class OrderFactory(DjangoModelFactory):
    class Meta:
        model = Order

    user = factory.SubFactory(UserFactory)
    status = "new"
    total = factory.fuzzy.FuzzyDecimal(10.00, 500.00, precision=2)
```

---

## 7. Test-Driven Development

Test-Driven Development (TDD) inverts the traditional workflow: tests are written *before* the code they test.

### 7.1 The Red-Green-Refactor Cycle

```
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
    RED: Write a failing test                         │
    (the test describes desired behavior)             │
        │                                             │
        ▼                                             │
    GREEN: Write the minimum code to pass the test    │
    (no more, no less)                                │
        │                                             │
        ▼                                             │
    REFACTOR: Clean up the code                       │
    (the test suite ensures you didn't break          │
     anything)                                        │
        │                                             │
        └─────────────────────────────────────────────┘
```

### 7.2 TDD Benefits and Costs

| Benefit | Explanation |
|---------|-------------|
| Tests as specification | Tests document exactly what the code should do |
| Design pressure | Hard-to-test code usually has poor design; TDD reveals this early |
| Confidence to refactor | Green test suite proves refactoring didn't break anything |
| Regression safety net | Every bug fix gets a test that prevents recurrence |

| Cost | Mitigation |
|------|-----------|
| Slower initial development | Offset by reduced debugging time and higher quality |
| Learning curve | Team training; pair programming with experienced practitioners |
| UI/integration tests harder | Apply TDD at the unit level; use separate integration test strategy |

### 7.3 Brief Example

```python
# Step 1: RED — write a failing test
def test_fizzbuzz_returns_fizz_for_multiples_of_3():
    assert fizzbuzz(3) == "Fizz"
    assert fizzbuzz(6) == "Fizz"
    assert fizzbuzz(9) == "Fizz"

# Step 2: GREEN — minimum code to pass
def fizzbuzz(n):
    if n % 3 == 0:
        return "Fizz"
    return str(n)

# Step 3: Add more tests → RED
def test_fizzbuzz_returns_buzz_for_multiples_of_5():
    assert fizzbuzz(5) == "Buzz"

# Step 4: GREEN
def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
```

TDD is covered in greater depth in the Programming topic (Lesson 10: Testing and TDD).

---

## 8. Regression Testing and Test Automation

### 8.1 Regression Testing

A regression is a bug introduced by a change that used to work correctly. Regression testing re-runs previously passing tests after a change to detect regressions.

**The regression trap**: As systems grow, the manual regression test suite becomes impossibly large. A 100-feature system with 50 tests each = 5,000 manual test executions per release — infeasible.

**Solution**: Automate the regression suite. Every bug fixed gets an automated test. Every feature gets automated tests. The suite runs on every commit.

### 8.2 The Test Automation Pyramid

The test pyramid (Mike Cohn) prescribes the right balance of test types:

```
                     ╱╲
                    ╱  ╲
                   ╱ UI ╲
                  ╱ Tests ╲   ← Slow, brittle, expensive
                 ╱──────────╲     Few (10–20% of suite)
                ╱            ╲
               ╱  Integration  ╲
              ╱     Tests        ╲  ← Medium speed/cost
             ╱────────────────────╲    Some (20–30% of suite)
            ╱                      ╲
           ╱      Unit Tests         ╲  ← Fast, cheap, precise
          ╱────────────────────────────╲    Many (50–70% of suite)
```

Inverting the pyramid (many UI tests, few unit tests) produces a slow, brittle test suite that gives poor feedback and is expensive to maintain.

### 8.3 CI/CD Integration

```yaml
# GitHub Actions: automated test pipeline
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run unit tests with coverage
        run: |
          pytest tests/unit/ -v \
            --cov=src \
            --cov-report=xml \
            --cov-fail-under=85
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests  # only run if unit tests pass
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://postgres:testpass@localhost/testdb
```

---

## 9. Reviews and Inspections

Static V&V (examining artifacts without execution) is often more cost-effective than testing for finding certain classes of defects.

### 9.1 The Fagan Inspection

Developed by Michael Fagan at IBM in 1976. Formal, role-based, and highly effective: Fagan's original study reported finding 82% of defects before the first test run.

**Roles**:
| Role | Responsibility |
|------|----------------|
| **Moderator** | Plans and facilitates; ensures process is followed |
| **Author** | Created the artifact under review; answers questions |
| **Reader** | Reads/paraphrases the artifact during inspection |
| **Inspector(s)** | Find defects; prepare independently before meeting |
| **Scribe** | Records all defects and decisions |

**Phases**:
1. **Planning**: Moderator checks entry criteria; distributes materials
2. **Overview**: Author explains context and design intent
3. **Preparation**: Each inspector reviews independently, logs issues
4. **Inspection Meeting**: Reader reads; inspectors raise issues; scribe records
5. **Rework**: Author fixes all logged defects
6. **Follow-up**: Moderator verifies all defects addressed; checks exit criteria

**Optimal inspection rate**: 100–200 lines of code per hour. Rushing produces negligible results.

### 9.2 Walkthroughs

Less formal than Fagan inspection. The author presents the artifact and walks the team through it, inviting questions and comments. Good for education and knowledge transfer. Less systematic than formal inspection.

### 9.3 Peer Code Review (Pull Request Review)

The most common form of code review in modern teams. Covered in detail in Lesson 07 (Software Quality Assurance, Section 10).

### 9.4 Inspection vs Testing Effectiveness

| Technique | Best at finding | Not good at finding |
|-----------|-----------------|---------------------|
| Inspection | Logic errors, design problems, standards violations, missing requirements | Timing bugs, performance problems |
| Testing | Integration failures, performance issues, timing bugs | Missing functionality (can't test what isn't there) |

They are complementary. Neither alone is sufficient.

---

## 10. Formal Verification

Formal verification uses mathematical proof to establish that a system satisfies its specification. It provides the highest assurance level but is expensive and requires specialized expertise.

### 10.1 Model Checking

Model checking exhaustively explores all possible states of a finite-state model of the system. It either confirms a property holds for all states or produces a counterexample trace.

**Use cases**: Communication protocols (TLS handshake), concurrent systems (race condition checking), hardware design.

**Tools**: SPIN, TLA+, Alloy, NuSMV.

```tla
(* TLA+ specification: a simple mutual exclusion algorithm *)
VARIABLES pc1, pc2, flag1, flag2

Init == pc1 = "start" /\ pc2 = "start"
     /\ flag1 = FALSE /\ flag2 = FALSE

(* Process 1 sets its flag and waits for process 2 to clear *)
Next1 == \/ /\ pc1 = "start"    /\ pc1' = "try"    /\ flag1' = TRUE  /\ UNCHANGED <<pc2, flag2>>
         \/ /\ pc1 = "try"     /\ ~flag2            /\ pc1' = "cs"    /\ UNCHANGED <<flag1, pc2, flag2>>
         \/ /\ pc1 = "cs"      /\ pc1' = "start"   /\ flag1' = FALSE  /\ UNCHANGED <<pc2, flag2>>

(* Safety: both processes cannot be in the critical section simultaneously *)
MutualExclusion == ~(pc1 = "cs" /\ pc2 = "cs")
```

### 10.2 Theorem Proving

Theorem provers (Coq, Isabelle/HOL, Lean) require human guidance to construct proofs. They can handle infinite state spaces that model checkers cannot.

**Notable applications**:
- seL4 microkernel: first OS kernel with complete formal correctness proof
- CompCert C compiler: formally verified to produce correct machine code from C
- CryptoVerif: formal verification of cryptographic protocols

### 10.3 Where Formal Methods Apply

```
Assurance     Cost     Typical domain
Level         Factor
────────────────────────────────────────────────
Testing       1x       All software
Code review   1.5x     All software
Fagan insp.   3x       High-reliability systems
Model check.  5–10x    Protocols, concurrent systems
Theorem prov. 20–50x   Safety-critical, cryptographic
```

Formal methods are practical for:
- Small, well-defined components (a scheduler, a protocol state machine)
- Security-critical algorithms (cryptographic primitives)
- Systems where failure cost is extreme (pacemakers, aircraft flight control)

---

## 11. The Bug Lifecycle

### 11.1 Bug States

```
         Discovered
              │
              ▼
           [New] ─── duplicate? ──▶ [Duplicate] → closed
              │
              ▼
          [Assigned] ─── not a bug? ──▶ [Rejected] → closed
              │
          developer
           works on
              │
              ▼
           [Fixed]
              │
           tester
          verifies
              ├──── still fails ──▶ [Reopened] ──▶ [Assigned]
              │
              ▼
          [Verified]
              │
              ▼
           [Closed]
```

### 11.2 Bug Severity vs Priority

| Severity | Definition | Example |
|----------|------------|---------|
| **Critical** | System crash, data loss, security breach — no workaround | Login always crashes the app |
| **High** | Major feature broken, no workaround | Users cannot complete checkout |
| **Medium** | Feature broken but workaround exists | Export to PDF fails; users can copy-paste |
| **Low** | Minor issue; cosmetic | Button misaligned by 2px |

| Priority | Definition | Example |
|----------|------------|---------|
| **P1** | Fix immediately — stop the release or rollback | Critical bug in payment processing |
| **P2** | Fix in current sprint | High-severity bug blocking key user workflow |
| **P3** | Fix in next sprint | Medium bug that has a known workaround |
| **P4** | Fix when time allows | Low-severity cosmetic bug |

Severity and priority are independent. A critical bug in a rarely-used admin feature may be P3. A low-severity bug on the landing page seen by all users may be P2.

### 11.3 Writing an Effective Bug Report

```
Title:       Password reset link expires after 1 use (expected: 10 minutes)
ID:          BUG-2847
Severity:    High
Priority:    P2
Reporter:    Q. Chen
Assigned to: R. Patel
Version:     v2.3.1
Environment: Production (also reproduced on staging)

Steps to Reproduce:
  1. Click "Forgot Password" on login page
  2. Enter registered email address
  3. Check email; click the reset link → redirected to reset form ✓
  4. Set new password → success message ✓
  5. Within 10 minutes, click the same link from the email again

Expected:
  Form displays "This link has expired" (link should be valid for 10 minutes)

Actual:
  Server returns HTTP 500 Internal Server Error

Attachments:
  - screenshot_500_error.png
  - server_error_log_2024-03-15_14:22:07.txt

Additional context:
  Only happens when the same link is used a second time. The 500 suggests
  the token deletion throws an exception if the token is already deleted.
  Possible missing "token exists" check before deletion.
```

---

## 12. Summary

Verification and Validation form a complementary system: verification ensures the software is built correctly (according to specification); validation ensures the right product is being built (meets user needs).

Key takeaways:

- **Four testing levels** — unit, integration, system, acceptance — each with a distinct scope and purpose. Design your test strategy to address all four.
- **Black-box techniques** — equivalence partitioning, boundary value analysis, decision tables, state transition — provide systematic coverage of specification-level behavior without requiring access to source code.
- **White-box techniques** — statement, branch, path, condition coverage — provide quantitative measures of structural thoroughness. Aim for ≥80% branch coverage as a practical minimum.
- **The test pyramid** — many fast unit tests, fewer integration tests, few end-to-end tests — produces a fast, maintainable test suite. Inverting it creates a slow, brittle one.
- **Reviews and inspections** find different defects than testing; formal Fagan inspection is among the most cost-effective defect-finding techniques ever measured.
- **TDD** integrates test writing into the development rhythm and produces a regression suite as a free by-product.
- **Formal verification** provides mathematical certainty at high cost; appropriate for safety-critical and security-critical components.
- **Bug reports** are communication artifacts: complete, reproducible, specific reports get fixed faster.

---

## 13. Practice Exercises

**Exercise 1 — Equivalence Partitioning and Boundary Value Analysis**

A password validation function requires:
- Length: 8–64 characters
- Must contain at least one uppercase letter
- Must contain at least one digit
- Must contain at least one special character from `!@#$%^&*()`
- Must not contain spaces

(a) Identify all equivalence partitions for each rule.
(b) Create a boundary value test set for the length rule.
(c) Using a decision table, identify all combinations of condition violations and the expected error message for each.

**Exercise 2 — Coverage Analysis**

For the following function, draw the control flow graph. Then:
(a) Calculate the cyclomatic complexity.
(b) Identify a minimal test set that achieves 100% branch coverage.
(c) Identify all independent paths (for path coverage). How many test cases does path coverage require?

```python
def shipping_cost(weight_kg, express, country):
    if weight_kg <= 0:
        raise ValueError("Weight must be positive")
    base = weight_kg * 2.50
    if express:
        base *= 1.75
    if country == "domestic":
        return base
    elif country == "canada":
        return base * 1.20
    else:
        return base * 2.00
```

**Exercise 3 — Test Plan**

You are testing a mobile banking application before its v2.0 release. The release includes:
- New biometric authentication (fingerprint / Face ID)
- Peer-to-peer payment feature (send money to contacts)
- Redesigned transaction history screen

Write a test plan outline including:
- Scope (what is in and out of scope)
- Testing approach (which levels and types of tests, and why)
- Entry and exit criteria
- At least 3 risks and their mitigation strategies

**Exercise 4 — Fagan Inspection**

Your team is about to do a Fagan inspection of a 150-line authentication module. The module was written by a senior developer and will be reviewed by four engineers.

(a) How long should the preparation phase and the inspection meeting each take, given recommended inspection rates?
(b) Design an inspection checklist with at least 8 items specific to authentication code.
(c) During the inspection, the author starts explaining why each design decision was made before inspectors raise issues. What is the moderator's responsibility here, and why?

**Exercise 5 — Bug Report**

You discover the following behavior in a web application: when you sort the product list by "Price: Low to High," items priced at $0.00 (free) appear at the bottom of the list, not the top.

Write a complete, professional bug report following the template from Section 11.3. Include:
- A descriptive title
- Appropriate severity and priority ratings with justification
- Precise steps to reproduce
- Expected vs actual behavior
- At least one hypothesis about the root cause

---

## 14. Further Reading

- **Books**:
  - *The Art of Software Testing* (3rd ed.) — Glenford Myers, Corey Sandler, Tom Badgett. The classic introduction.
  - *Software Testing: A Craftsman's Approach* (4th ed.) — Paul Jorgensen. Comprehensive coverage of all testing techniques.
  - *Continuous Delivery* — Jez Humble and David Farley. How to automate the entire delivery pipeline including testing.
  - *Introduction to the Theory of Computation* — Michael Sipser. Background for formal methods.

- **Standards**:
  - IEEE Std 829-2008 — IEEE Standard for Software and System Test Documentation
  - IEEE Std 1012-2016 — IEEE Standard for System, Software, and Hardware Verification and Validation
  - ISO/IEC 29119 — Software testing standard (5-part series)

- **Tools**:
  - pytest — https://pytest.org/ (Python testing framework)
  - Jest — https://jestjs.io/ (JavaScript testing)
  - Locust — https://locust.io/ (load testing)
  - SPIN — http://spinroot.com/ (model checker for concurrent systems)
  - TLA+ — https://lamport.azurewebsites.net/tla/tla.html (formal specification language)

- **Papers and Articles**:
  - Fagan, M. E. (1976). "Design and Code Inspections to Reduce Errors in Program Development." *IBM Systems Journal*.
  - Myers, G. J. (1978). "A Controlled Experiment in Program Testing and Code Walkthroughs/Inspections." *Communications of the ACM*.
  - Boehm, B. (1979). "Guidelines for Verifying and Validating Software Requirements and Design Specifications." *EURO IFIP*.

---

## Exercises

### Exercise 1: Design Black-Box Test Cases

A discount calculation function has the following specification:

- Input: `cart_total` (float, must be > 0) and `coupon_code` (string, optional)
- Valid coupon codes: `"SAVE10"` (10% off), `"SAVE20"` (20% off), `"FREESHIP"` (no price change, free shipping flag set)
- If `cart_total` < 10.00, no discount is applied even with a valid coupon
- If `coupon_code` is None or empty, no discount is applied
- Invalid coupon codes return an error

Using equivalence partitioning and boundary value analysis, derive a minimal test set. For each test case, specify inputs, the equivalence partition it represents, and the expected output.

### Exercise 2: Achieve Branch Coverage

For the function below, draw the control flow graph and identify a minimal set of test cases that achieves 100% branch coverage. State each branch you must exercise.

```python
def classify_bmi(weight_kg, height_m):
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive")
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    else:
        return "obese"
```

(a) How many branches exist? List them.
(b) What is the minimum number of test cases needed for 100% branch coverage?
(c) Would this test set also achieve 100% path coverage? Why or why not?

### Exercise 3: Plan Integration Testing

A microservices order system has these services: `OrderService`, `InventoryService`, `PaymentService`, `NotificationService`. `OrderService` calls `InventoryService` and `PaymentService`; `NotificationService` is called by `OrderService` after a successful payment.

(a) Sketch the integration dependency graph.
(b) Design an incremental bottom-up integration test plan: specify which services are integrated in each phase, what stubs/drivers are needed, and what interface contracts each phase verifies.
(c) Identify three integration-specific failure modes (not unit-level bugs) that could only be caught at integration level.

### Exercise 4: Write a Test Plan for a New Feature

A mobile banking app is adding biometric login (fingerprint and Face ID) as an alternative to PIN entry. Write a one-page test plan outline covering:

- Scope: what is in and out of scope for this feature's test cycle
- Test levels: which levels (unit, integration, system, acceptance) apply and what each tests
- Testing types: at least four non-functional types to consider and why each matters here
- Entry and exit criteria: specific, measurable conditions
- Top three risks to the test effort and a mitigation strategy for each

### Exercise 5: Analyze the Bug Lifecycle

A tester discovers that when a user attempts to export a 10,000-row report to PDF, the application returns a 504 Gateway Timeout after 32 seconds instead of generating the file.

(a) Write a complete bug report following the template from Section 11.3.
(b) Assign a severity and priority rating. Justify both independently.
(c) Propose two hypotheses about the root cause (one at the application layer, one at the infrastructure layer).
(d) Describe the verification steps the tester must perform after a developer marks the bug as Fixed.

---

**Previous**: [07. Software Quality Assurance](./07_Software_Quality_Assurance.md) | **Next**: [09. Configuration Management](./09_Configuration_Management.md)
