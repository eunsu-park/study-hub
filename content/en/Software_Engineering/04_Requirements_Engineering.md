# Lesson 04: Requirements Engineering

**Previous**: [03. Agile and Iterative Development](./03_Agile.md) | **Next**: [05. Software Modeling and UML](./05_Software_Modeling_and_UML.md)

---

Requirements engineering is the foundation on which every software project stands or falls. Studies consistently show that errors introduced during requirements — misunderstood features, missing constraints, contradictory goals — are ten to one hundred times more expensive to fix once a system is deployed than they would have been to catch at the start. This lesson covers the full lifecycle of requirements: discovering what stakeholders actually need, writing those needs down precisely, validating them, and keeping them under control as the project evolves.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Lesson 01 — What Is Software Engineering
- Lesson 02 — Software Development Life Cycles
- Lesson 03 — Agile and Iterative Development

**Learning Objectives**:
- Distinguish functional requirements from non-functional requirements and constraints
- Apply elicitation techniques appropriate to a given stakeholder situation
- Write requirements that satisfy the SMART and IEEE 830 / ISO 29148 quality criteria
- Produce a well-structured Software Requirements Specification (SRS) document
- Write user stories with acceptance criteria in Given/When/Then format
- Build and use a Requirements Traceability Matrix (RTM)
- Prioritize requirements using MoSCoW and the Kano model
- Explain strategies for managing requirements changes throughout a project

---

## Table of Contents

1. [What Are Requirements?](#1-what-are-requirements)
2. [The Requirements Engineering Process](#2-the-requirements-engineering-process)
3. [Elicitation Techniques](#3-elicitation-techniques)
4. [Writing Good Requirements](#4-writing-good-requirements)
5. [Software Requirements Specification](#5-software-requirements-specification)
6. [User Stories and Acceptance Criteria](#6-user-stories-and-acceptance-criteria)
7. [Requirements Traceability](#7-requirements-traceability)
8. [Requirements Prioritization](#8-requirements-prioritization)
9. [Managing Changing Requirements](#9-managing-changing-requirements)
10. [Tooling](#10-tooling)
11. [Summary](#11-summary)
12. [Practice Exercises](#12-practice-exercises)
13. [Further Reading](#13-further-reading)

---

## 1. What Are Requirements?

A **requirement** is a statement of what a system must do or a quality it must possess in order to satisfy its stakeholders. Requirements form the contract between the people who need the software and the people who build it. Getting them right is one of the most difficult intellectual tasks in engineering because requirements bridge the gap between human intent — which is fuzzy, context-dependent, and often partially unconscious — and computational specification, which must be unambiguous and verifiable.

### 1.1 Functional Requirements

Functional requirements (FRs) describe the **behaviors** the system shall exhibit: the inputs it accepts, the computations it performs, and the outputs it produces.

Examples:
- "The system shall allow a registered user to reset their password via a six-digit code sent to their registered email address."
- "The payment service shall complete a credit card authorization in under two seconds for 99% of transactions."

Functional requirements answer the question: *What shall the system do?*

### 1.2 Non-Functional Requirements

Non-functional requirements (NFRs) — also called **quality attributes** or **system qualities** — describe *how well* the system performs its functions. They constrain the solution space without dictating a particular function.

| Category | Examples |
|---|---|
| **Performance** | Response time, throughput, latency |
| **Reliability** | MTBF, availability (e.g., 99.9% uptime) |
| **Security** | Authentication, authorization, data encryption |
| **Usability** | Learnability, accessibility (WCAG 2.1 AA) |
| **Maintainability** | Modularity, testability, technical debt limits |
| **Scalability** | Concurrent users, horizontal/vertical scaling |
| **Portability** | Supported operating systems, browsers |
| **Compliance** | GDPR, HIPAA, PCI-DSS |

NFRs are often more architecturally significant than functional requirements because they cut across the entire system. A requirement that "the system shall be available 99.99% of the time" forces redundancy, failover mechanisms, and monitoring at every layer.

### 1.3 Constraints

Constraints are requirements that eliminate design choices without being strictly functional or non-functional:

- "The system shall be implemented in Python 3.11 or later."
- "The system shall deploy on the customer's on-premise VMware cluster."
- "The project shall deliver an initial release within six calendar months."

### 1.4 The Requirements Hierarchy

Requirements exist at multiple levels of abstraction:

```
Business Requirements    (WHY — vision, goals, ROI)
        |
User Requirements        (WHO/WHAT — user goals, use cases)
        |
System Requirements      (WHAT — detailed functional + NFR)
        |
Design Constraints       (HOW — technology, platform choices)
```

A common source of project failure is treating system-level requirements as if they were business goals, or skipping straight to design constraints before user needs are understood.

---

## 2. The Requirements Engineering Process

Requirements engineering (RE) is not a single phase but an ongoing process with five tightly coupled activities.

```
  ┌─────────────┐
  │ Elicitation │  ← discover stakeholder needs
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Analysis   │  ← resolve conflicts, model, prioritize
  └──────┬──────┘
         │
  ┌──────▼───────────┐
  │  Specification   │  ← write precise, verifiable requirements
  └──────┬───────────┘
         │
  ┌──────▼──────┐
  │  Validation │  ← confirm requirements are correct & complete
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │ Management  │  ← track, change-control, trace requirements
  └─────────────┘
```

In practice, these activities iterate throughout the project. Even on a waterfall project, requirements creep back in during design and testing.

### 2.1 Elicitation

Elicitation is the process of **discovering** requirements from stakeholders and other sources. It is the most human-intensive phase and the one most likely to introduce gaps.

Key challenges:
- Stakeholders don't know what they want until they see something that is wrong
- Domain experts know their domain but not what technology can or cannot do
- Different stakeholders have conflicting goals
- Tacit knowledge (things people "just know") is rarely articulated spontaneously

### 2.2 Analysis

During analysis, engineers:
- Detect and resolve conflicts between stakeholder requirements
- Identify overlapping or duplicate requirements
- Check for feasibility (technical, financial, schedule)
- Build models (use cases, data flow diagrams, entity-relationship diagrams) to reason about the system

### 2.3 Specification

Specification produces documented artifacts — the Software Requirements Specification (SRS), user story backlog, or equivalent — in which requirements are stated precisely enough to be verified.

### 2.4 Validation

Validation asks: *are these the right requirements?* It is distinct from verification, which asks whether the system implements the requirements correctly. Validation techniques include:

- Structured walkthroughs and reviews
- Prototyping
- Test case derivation (if you can't write a test, the requirement may be too vague)
- Formal inspections (Fagan inspection)

### 2.5 Management

Requirements management covers:
- Storing requirements in a controlled repository
- Tracking status (proposed, approved, implemented, tested, obsolete)
- Handling change requests through a formal change-control board (CCB)
- Maintaining traceability links

---

## 3. Elicitation Techniques

No single technique works for all projects or all stakeholder groups. Skilled requirements engineers combine several.

### 3.1 Interviews

One-on-one or small-group conversations with stakeholders. Interviews can be:

- **Structured**: fixed set of questions, good for consistency across many stakeholders
- **Unstructured**: open conversation, good for discovering unexpected concerns
- **Semi-structured**: a guide of topics with flexibility for follow-up

**Tips**:
- Ask open-ended questions: "Walk me through your typical workday" rather than "Do you need a dashboard?"
- Use the Five Whys to dig below stated wants to real needs
- Record (with permission) and transcribe; memory is unreliable

### 3.2 Questionnaires and Surveys

When stakeholders are numerous or geographically distributed, questionnaires scale where interviews cannot. They are best used to:

- Quantify frequency of existing activities ("How often do you generate this report?")
- Rank preferences among known alternatives
- Validate conclusions from interviews against a broader population

Limitation: surveys cannot probe unexpected answers.

### 3.3 Observation (Ethnography)

The analyst shadows users in their natural work environment without interfering. This surfaces tacit knowledge that users cannot articulate in an interview because they have automated it:

- "I always check the fax machine before entering orders" (a step not documented anywhere)
- The actual sequence in which screens are used (which may differ from the intended sequence)

### 3.4 Workshops (JAD / Requirements Workshops)

Joint Application Development (JAD) sessions bring stakeholders, developers, and a facilitator together for one to several days. Benefits:

- Decisions are made in real time with all parties present
- Conflicts surface and get resolved immediately
- Shared understanding forms quickly

Require strong facilitation skills; a poorly run workshop can polarize stakeholders.

### 3.5 Prototyping

A low-fidelity prototype (paper sketches, wireframes, or a click-through mockup) helps stakeholders articulate what they want by reacting to something concrete. This is especially effective for UI-heavy applications.

Types:
- **Throwaway (exploratory)**: built only to elicit requirements, then discarded
- **Evolutionary**: refined and eventually becomes the actual system

Risk: stakeholders may assume the prototype represents the final product timeline.

### 3.6 Use Cases

A **use case** is a scenario describing how a user (actor) interacts with the system to achieve a goal. Use cases are both an elicitation technique (drawn collaboratively with stakeholders) and a specification artifact. They capture functional requirements in a narrative that is accessible to non-technical stakeholders.

### 3.7 Document Analysis

Review existing documentation: current system manuals, business process descriptions, regulatory standards, competitor products, industry standards. Useful when replacing a legacy system.

### 3.8 Brainstorming

Facilitated ideation session where participants generate ideas without immediate criticism. Useful early to expand the solution space before narrowing down to requirements.

---

## 4. Writing Good Requirements

A requirement that is ambiguous, untestable, or contradictory is worse than no requirement at all — it creates a false sense that something is decided.

### 4.1 The SMART Criteria

| Letter | Meaning | Question to ask |
|---|---|---|
| **S** | Specific | Is it clear exactly what is required? |
| **M** | Measurable | Can we verify that it has been met? |
| **A** | Achievable | Is it technically and economically feasible? |
| **R** | Relevant | Does it trace to a real stakeholder need? |
| **T** | Time-bound | Is there a deadline or version target? |

### 4.2 Common Requirement Defects

| Defect | Example | Fix |
|---|---|---|
| Ambiguous | "The system shall respond quickly" | Specify "< 200 ms for 95th percentile" |
| Non-verifiable | "The system shall be user-friendly" | Define usability metric (e.g., task completion rate > 90%) |
| Compound | "The system shall log and archive all events" | Split into two separate requirements |
| Implies design | "The system shall use a relational database" | State the need: "data shall be persistent and queryable" |
| Passive voice trap | "Errors shall be handled" | "The system shall display an error message to the user within 1 second of detecting a validation failure" |
| Unbounded | "The system shall support all browsers" | List specific supported browsers and minimum versions |

### 4.3 IEEE 830 / ISO 29148 Quality Attributes

The IEEE 830 standard (now superseded by ISO/IEC/IEEE 29148:2018) defines quality attributes for individual requirements:

- **Correct**: accurately reflects stakeholder needs
- **Unambiguous**: only one interpretation
- **Complete**: no missing information required to implement or test
- **Consistent**: no contradictions with other requirements
- **Ranked** (for importance/stability): priority and volatility are known
- **Verifiable**: at least one test exists that can confirm the requirement is met
- **Modifiable**: structured so changes can be made consistently
- **Traceable**: origin is known; it can be traced forward to design and tests

### 4.4 Language Conventions

Use modal verbs consistently:
- **Shall** — mandatory requirement
- **Should** — recommendation (desirable but not mandatory)
- **May** — optional
- **Will** — statement of fact about the world (not a requirement)

Every requirement should reference a single, identifiable **subject** (who or what must do something):

```
BAD:  "Passwords must be at least eight characters."
GOOD: "The system shall reject any password shorter than eight characters
       and display the message: 'Password must be at least 8 characters.'"
```

---

## 5. Software Requirements Specification

The Software Requirements Specification (SRS) is the primary artifact of requirements engineering. IEEE 830 provides a standard structure; ISO/IEC/IEEE 29148 refines it.

### 5.1 Standard SRS Structure

```
1. Introduction
   1.1 Purpose
   1.2 Scope
   1.3 Definitions, Acronyms, Abbreviations
   1.4 References
   1.5 Overview

2. Overall Description
   2.1 Product Perspective
   2.2 Product Functions (high-level summary)
   2.3 User Classes and Characteristics
   2.4 Operating Environment
   2.5 Design and Implementation Constraints
   2.6 User Documentation
   2.7 Assumptions and Dependencies

3. System Features (Functional Requirements)
   3.x Feature Name
      3.x.1 Description and Priority
      3.x.2 Stimulus/Response Sequences
      3.x.3 Functional Requirements

4. External Interface Requirements
   4.1 User Interfaces
   4.2 Hardware Interfaces
   4.3 Software Interfaces
   4.4 Communications Interfaces

5. Non-Functional Requirements
   5.1 Performance Requirements
   5.2 Safety Requirements
   5.3 Security Requirements
   5.4 Software Quality Attributes
   5.5 Business Rules

6. Other Requirements

Appendices
Index
```

### 5.2 Writing the SRS in Practice

In agile contexts the SRS is often replaced by or complemented with:
- A **product backlog** of user stories (the living, prioritized list)
- A **product vision** document (the "why" and "for whom")
- **Definition of Done** and **acceptance criteria** on each story

However, for safety-critical, regulatory, or contractual software (medical devices, avionics, government procurement), a formal SRS remains legally and technically necessary.

### 5.3 Requirement Identifiers

Every requirement must have a unique, stable identifier to support traceability and change control:

```
SRS-FUNC-AUTH-001: The system shall authenticate users via username and password.
SRS-FUNC-AUTH-002: The system shall lock an account after five consecutive failed login attempts.
SRS-NFR-PERF-001: The authentication response shall complete within 500 ms for 99% of requests under normal load.
```

A common naming convention: `[DOC]-[TYPE]-[SUBSYSTEM]-[SEQ]`

---

## 6. User Stories and Acceptance Criteria

In agile development, requirements are often expressed as **user stories** — short, informal descriptions of a feature from the perspective of a user.

### 6.1 The User Story Template

```
As a [role],
I want [feature/capability],
so that [benefit/value].
```

Examples:

```
As a registered customer,
I want to save my shipping address to my account,
so that I do not have to re-enter it on future orders.

As a system administrator,
I want to receive an email alert when disk usage exceeds 80%,
so that I can take preventive action before storage runs out.
```

### 6.2 INVEST Criteria for Good User Stories

| Letter | Meaning |
|---|---|
| **I** | Independent — can be developed in any order |
| **N** | Negotiable — details can change through conversation |
| **V** | Valuable — delivers value to a stakeholder |
| **E** | Estimable — team can estimate the effort |
| **S** | Small — fits within one sprint |
| **T** | Testable — acceptance criteria can be written |

### 6.3 Acceptance Criteria

Acceptance criteria define the conditions under which a story is considered done. The **Given/When/Then (Gherkin)** format makes them automatable:

```gherkin
Feature: Password Reset

  Scenario: Successful password reset via email
    Given a registered user with email "alice@example.com"
    And the user is on the "Forgot Password" page
    When the user enters "alice@example.com" and submits
    Then the system sends a reset code to "alice@example.com"
    And the user is shown the message "Check your email for a reset code"

  Scenario: Invalid email address
    Given a user is on the "Forgot Password" page
    When the user enters "notregistered@example.com" and submits
    Then the system shows the same success message
    And no email is sent
    # (security: do not reveal whether email exists)

  Scenario: Expired reset code
    Given a user received a reset code more than 30 minutes ago
    When the user submits the expired code
    Then the system displays "This code has expired. Request a new one."
    And the user is redirected to the "Forgot Password" page
```

Gherkin scenarios double as the specification for automated acceptance tests (e.g., Cucumber, Behave, SpecFlow).

### 6.4 Epics and Story Splitting

A **story** should fit in one sprint (one to two weeks). Larger items are called **epics** and must be split. Common splitting patterns:

| Pattern | Example |
|---|---|
| By user role | "Admin manages users" → separate stories for create, edit, deactivate |
| By data type | "Export report" → CSV, PDF, Excel as separate stories |
| By workflow step | "Complete checkout" → add to cart, enter shipping, enter payment, confirm |
| By happy/unhappy path | Success path first; error handling as a follow-on story |
| By performance | Functional story first, then a performance NFR story |

---

## 7. Requirements Traceability

A **Requirements Traceability Matrix (RTM)** is a table that links requirements to their sources and to the artifacts that implement and verify them.

### 7.1 Types of Traceability

- **Backward traceability**: requirement → source (stakeholder, regulation, business goal)
- **Forward traceability**: requirement → design element → code module → test case

### 7.2 RTM Example

| Req ID | Description | Source | Design Doc | Code Module | Test Case |
|---|---|---|---|---|---|
| FR-AUTH-001 | User login via email/password | Stakeholder interview (J. Smith) | ARCH-3.2 | auth/login.py | TC-AUTH-001 |
| FR-AUTH-002 | Account lockout after 5 failures | OWASP ASVS 2.2.1 | ARCH-3.3 | auth/lockout.py | TC-AUTH-005 |
| NFR-PERF-001 | Login < 500 ms (p99) | SLA with client | ARCH-7.1 | — | PT-001 |

### 7.3 Benefits of Traceability

- **Impact analysis**: when a requirement changes, immediately identify which designs, code, and tests are affected
- **Coverage analysis**: ensure every requirement has at least one test
- **Completeness checking**: ensure every design element traces back to a requirement (no gold-plating)
- **Regulatory compliance**: demonstrate that safety-critical requirements are implemented and tested

---

## 8. Requirements Prioritization

Not all requirements are equally important or urgent. Prioritization guides scope decisions, release planning, and trade-off negotiations.

### 8.1 MoSCoW Method

| Category | Meaning | Guidance |
|---|---|---|
| **Must have** | Non-negotiable; product fails without it | ~60% of effort; always in MVP |
| **Should have** | High value but not critical for launch | Include if time permits |
| **Could have** | Nice to have; low impact if omitted | Target for later releases |
| **Won't have** | Out of scope for this release | Formally excluded to manage expectations |

MoSCoW is simple and widely understood by non-technical stakeholders. The main risk is that every stakeholder wants their requirements in "Must have."

### 8.2 Kano Model

The Kano model classifies features by their effect on customer satisfaction:

| Category | Characteristic | Example |
|---|---|---|
| **Basic (Must-be)** | Expected; absence causes dissatisfaction but presence causes no delight | Login works; data is saved |
| **Performance (Linear)** | More is better; satisfaction scales with level | Faster load times, higher storage |
| **Excitement (Delighter)** | Unexpected; presence delights, absence is acceptable | Personalized recommendations |
| **Indifferent** | Customers don't care either way | Some UI chrome |
| **Reverse** | Presence causes dissatisfaction | Forced social sharing |

The Kano model helps teams avoid over-investing in performance attributes at the expense of unmet basic attributes.

### 8.3 Weighted Scoring

Assign each requirement a score based on weighted criteria:

```
Score = (Business Value × 0.4) + (Strategic Fit × 0.3) + (Risk Reduction × 0.2) + (Customer Request Frequency × 0.1)
```

Requirements are then ranked by score. This is more rigorous than MoSCoW and useful when ROI analysis is needed.

### 8.4 Value vs. Effort Matrix

Plot requirements on a 2×2 grid:

```
High Value │ Quick Wins ★    │ Major Projects
           │                │
Low Value  │ Fill-Ins       │ Thankless Tasks ✗
           └────────────────┴──────────────
              Low Effort        High Effort
```

Prioritize Quick Wins; plan Major Projects; schedule Fill-Ins; avoid Thankless Tasks.

---

## 9. Managing Changing Requirements

Requirements change is inevitable — not a failure of process. Studies show 25–50% of requirements change before a project completes. The goal is to manage change, not prevent it.

### 9.1 Causes of Requirements Change

- Stakeholders gain clarity only when they see working software
- The business environment evolves (new regulation, competitor moves, market feedback)
- Technical discoveries reveal infeasibility of original requirements
- Personnel changes — new stakeholders bring new priorities

### 9.2 Change Control Process

```
1. Change Request submitted (by anyone)
   ↓
2. Impact Analysis (requirements engineer + architect)
   → Affected requirements, design, code, tests
   → Schedule and cost impact
   ↓
3. Change Control Board (CCB) review
   → Approve / Reject / Defer
   ↓
4. If approved: update SRS, update RTM, notify team
   ↓
5. Implement and verify change
```

### 9.3 Strategies for Dealing with Volatility

- **Defer volatile requirements**: place unstable items in a "parking lot" backlog for later releases
- **Design for extensibility**: architect the system so volatile areas can change with minimal ripple
- **Incremental delivery**: release working software frequently; each release flushes out misunderstandings before they compound
- **Requirements workshops at sprint boundaries**: review and re-prioritize regularly

### 9.4 Baseline and Versioning

A **requirements baseline** is a snapshot of requirements that has been formally agreed and placed under configuration control. Any change after baseline requires a formal change request. Baselines are named (e.g., "v1.0 baseline") and correspond to milestones (contract signing, design review, code freeze).

---

## 10. Tooling

| Tool | Type | Use Case |
|---|---|---|
| **Jira** | Issue tracker / story backlog | Agile user story management, sprint planning |
| **Azure DevOps** | ALM suite | Work items, boards, RTM integration |
| **IBM DOORS / DOORS Next** | Requirements management | Safety-critical, regulatory, large-scale projects |
| **Confluence** | Wiki / documentation | SRS authoring, stakeholder collaboration |
| **Figma / Balsamiq** | Prototyping / wireframing | UI requirements elicitation |
| **Cucumber / Behave** | BDD test automation | Executable acceptance criteria (Gherkin) |
| **ReqIF** | Exchange format | Interoperability between RE tools (automotive, aerospace) |

---

## 11. Summary

Requirements engineering transforms vague stakeholder intentions into precise, verifiable specifications. The five core activities — elicitation, analysis, specification, validation, and management — repeat iteratively throughout a project.

Key takeaways:

- **Functional requirements** describe behaviors; **non-functional requirements** describe quality attributes. Both are necessary.
- Elicitation is a human activity requiring multiple techniques: interviews, observation, workshops, prototyping, and use cases.
- Good requirements are **specific, measurable, achievable, relevant, and time-bound** (SMART) and satisfy IEEE/ISO quality attributes: correct, unambiguous, complete, consistent, verifiable, and traceable.
- A formal **SRS** is structured according to IEEE 830 / ISO 29148; agile projects typically use a **product backlog** of user stories with Gherkin acceptance criteria.
- Every requirement must have a **unique identifier** and be linked in a **Requirements Traceability Matrix** to its source, design artifact, and test case.
- Prioritize using **MoSCoW** for quick consensus or the **Kano model** for understanding customer delight.
- Requirements **will** change; manage change through formal change requests, impact analysis, and baselining.

---

## 12. Practice Exercises

**Exercise 1: Classifying Requirements**

Classify each of the following as Functional (FR), Non-Functional (NFR), or Constraint (C), and identify any defects:

a. "The system should be fast."
b. "The application shall authenticate users via OAuth 2.0."
c. "The system shall be implemented using React 18."
d. "The checkout page shall load in under 1.5 seconds for 95% of users on a 4G connection."
e. "All data shall be encrypted and backed up daily."

---

**Exercise 2: Rewriting Bad Requirements**

Rewrite the following poorly formed requirements to meet SMART and ISO 29148 quality criteria:

a. "The system shall handle a lot of concurrent users."
b. "The UI shall be intuitive."
c. "The system shall use a database to store information."
d. "Errors should be handled gracefully."

---

**Exercise 3: User Stories and Acceptance Criteria**

You are building a library management system. Write three user stories (one for a library member, one for a librarian, one for an administrator) with at least two Gherkin acceptance criteria scenarios each. At least one scenario per story should cover an error or edge case.

---

**Exercise 4: Requirements Traceability Matrix**

Given the following requirements, design a minimal RTM:

- FR-001: Users can search books by title, author, or ISBN.
- FR-002: Users can place a hold on a checked-out book.
- NFR-001: Search results shall appear within 2 seconds for queries returning up to 1,000 results.

Add columns for: source, design reference, implementing module, and test case ID. Create at least one test case ID per requirement.

---

**Exercise 5: Prioritization**

You have the following candidate features for a ride-sharing app MVP. Use MoSCoW to categorize them, justifying your choices:

1. Ride booking from current location
2. Fare estimate before booking
3. In-app chat between driver and passenger
4. Driver ratings and reviews
5. Multi-stop rides
6. Ride history and receipts
7. Promotional discount codes
8. Real-time GPS tracking during ride
9. Scheduled future rides
10. Carbon offset tracking per ride

---

## 13. Further Reading

- **ISO/IEC/IEEE 29148:2018** — International standard for requirements engineering processes and artifacts
- Wiegers, K. & Beatty, J. — *Software Requirements* (3rd ed., Microsoft Press, 2013) — the most comprehensive practical guide
- Cockburn, A. — *Writing Effective Use Cases* (Addison-Wesley, 2000) — the classic use-case reference
- Cohn, M. — *User Stories Applied* (Addison-Wesley, 2004) — agile story practices
- Pohl, K. — *Requirements Engineering: Fundamentals, Principles, and Techniques* (Springer, 2010) — rigorous academic treatment
- IREB — *Certified Professional for Requirements Engineering (CPRE) Handbook* — certification body and reference glossary
- OWASP ASVS — *Application Security Verification Standard* — source of security NFRs

---

**Previous**: [03. Agile and Iterative Development](./03_Agile.md) | **Next**: [05. Software Modeling and UML](./05_Software_Modeling_and_UML.md)
