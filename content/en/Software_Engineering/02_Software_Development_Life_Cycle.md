# Lesson 2: Software Development Life Cycle

**Previous**: [What Is Software Engineering](./01_What_Is_Software_Engineering.md) | **Next**: [Agile and Iterative Development](./03_Agile_and_Iterative_Development.md)

---

Every software system passes through a predictable set of phases from its conception to its retirement. The **Software Development Life Cycle (SDLC)** is the structured process that defines these phases, the activities within each phase, the artifacts produced, and the criteria for moving from one phase to the next. Choosing an appropriate SDLC model is one of the most consequential decisions made at the start of a project.

**Difficulty**: ⭐⭐

**Prerequisites**:
- [Lesson 1: What Is Software Engineering](./01_What_Is_Software_Engineering.md)
- Basic familiarity with software projects

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the standard phases of the software development life cycle
2. Explain the Waterfall, V-Model, Incremental, Spiral, RAD, and Prototyping models
3. Identify the strengths, weaknesses, and appropriate use cases of each model
4. Apply a decision framework to choose a model for a given project
5. Understand how process models relate to risk, requirements stability, and team size

---

## 1. SDLC Overview

The Software Development Life Cycle describes the structure that guides software development from initial idea to decommissioning. Regardless of the specific model used, most SDLC frameworks include variations of the following core phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                   SDLC Core Phases                              │
│                                                                 │
│  1. Planning      ──►  What is the project? Is it feasible?    │
│  2. Requirements  ──►  What must the software do?              │
│  3. Design        ──►  How will it be structured?              │
│  4. Implementation──►  Write the code                          │
│  5. Testing       ──►  Verify it works correctly               │
│  6. Deployment    ──►  Release to production                   │
│  7. Maintenance   ──►  Fix defects, add features over time     │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Planning

**Goal**: Determine whether the project is worth doing and how it will be approached.

Activities:
- Define the project scope and objectives
- Identify stakeholders
- Perform a feasibility study (technical, financial, legal, operational)
- Estimate costs and schedule at a high level
- Define the team and its roles
- Select the SDLC model

Key artifacts: Project charter, feasibility report, initial project plan

### Phase 2: Requirements Engineering

**Goal**: Understand and document exactly what the software must do.

Activities:
- Elicit requirements through interviews, workshops, and observation
- Analyze and prioritize requirements
- Document functional and non-functional requirements
- Validate requirements with stakeholders
- Manage requirement changes

Key artifacts: Software Requirements Specification (SRS), use cases, user stories

### Phase 3: Design

**Goal**: Determine how the system will be built.

Activities:
- Architectural design (high-level structure: components, layers, interfaces)
- Detailed design (class structures, data schemas, algorithms)
- UI/UX design
- Database design
- Security design

Key artifacts: Architecture document, detailed design specifications, database schema, API contracts

### Phase 4: Implementation (Coding)

**Goal**: Translate the design into working code.

Activities:
- Write source code according to design specifications
- Code reviews
- Unit testing by developers
- Integration of components

Key artifacts: Source code, build artifacts, unit tests

### Phase 5: Testing

**Goal**: Verify that the software meets requirements and works correctly.

Activities:
- System testing (end-to-end verification)
- Integration testing
- Performance testing, security testing
- User Acceptance Testing (UAT)
- Defect reporting and resolution

Key artifacts: Test plans, test cases, defect reports, test results

### Phase 6: Deployment

**Goal**: Release the software to its intended environment and users.

Activities:
- Deployment planning and rollback procedures
- Production environment configuration
- User training and documentation
- Cutover from legacy systems (if applicable)

Key artifacts: Deployment plan, release notes, user manuals

### Phase 7: Maintenance and Evolution

**Goal**: Keep the software operational and evolving to meet changing needs.

Activities:
- Fix defects discovered in production (corrective maintenance)
- Adapt to new environments (adaptive maintenance)
- Add new features (perfective maintenance)
- Prevent future failures (preventive maintenance)

Key artifacts: Change requests, updated documentation, new releases

---

## 2. Waterfall Model

The Waterfall model, described by Winston Royce in his 1970 paper "Managing the Development of Large Software Systems," is the earliest formal SDLC model. Despite Royce actually arguing that pure sequential development was **risky** and recommending iterative approaches, the sequential interpretation became the dominant model for decades.

### Structure

```
     Requirements
          │
          ▼
        Design
          │
          ▼
    Implementation
          │
          ▼
       Testing
          │
          ▼
     Deployment
          │
          ▼
     Maintenance
```

Each phase must be **fully completed** and formally signed off before the next phase begins. Going back to a previous phase is possible but treated as an exceptional event.

### Phase Gate Criteria

| Phase Gate | Entry Criteria | Exit Criteria |
|------------|---------------|---------------|
| Requirements → Design | Project approved, team assembled | Signed-off SRS document |
| Design → Implementation | Approved SRS | Signed-off design documents |
| Implementation → Testing | Approved design | Code complete, unit tests passing |
| Testing → Deployment | Code complete | All critical defects resolved, UAT passed |
| Deployment → Maintenance | UAT sign-off | Production deployment complete |

### Strengths

- **Clarity**: The process is easy to understand and manage
- **Documentation**: Produces comprehensive documentation at each phase
- **Phase control**: Clear milestones make progress measurable
- **Works well for stable requirements**: If what needs to be built is well-understood and unlikely to change, sequential development is efficient
- **Vendor contracts**: Fixed-scope contracts with external vendors are easier to manage with Waterfall

### Weaknesses

- **Late discovery of defects**: Testing occurs late; defects found during testing are expensive to fix
- **No working software until late**: Stakeholders see nothing until testing or deployment
- **Poor fit for changing requirements**: Changes late in the process require expensive rework of earlier phases
- **Assumes perfect requirements**: In practice, requirements are almost never fully known at the start
- **Integration problems discovered late**: If components don't fit together, it's discovered during integration testing

### When to Use Waterfall

- Requirements are well-defined, stable, and unlikely to change
- The technology is well-understood (no major R&D)
- The project is short enough that changes are unlikely during development
- Regulatory or contractual requirements mandate comprehensive documentation
- Working with external contractors on fixed-scope deliverables

**Real-world example**: A government agency contracts a vendor to build a payroll system with legally mandated calculation rules. Requirements are defined by statute and unlikely to change during the 18-month project. Waterfall is appropriate.

---

## 3. V-Model (Verification and Validation Model)

The V-Model extends Waterfall by explicitly mapping each development phase to a corresponding testing phase. The left side of the V represents decomposition (breaking down requirements); the right side represents verification and validation.

### Structure

```
Requirements ──────────────────────────── Acceptance Testing
    │                                              │
    ▼                                              │
System Design ──────────────────── System Testing │
    │                                    │         │
    ▼                                    │         │
Architectural Design ──── Integration Testing     │
    │                              │              │
    ▼                              │              │
Module Design ── Unit Testing      │              │
    │                │             │              │
    └── Coding ──────┘             │              │
                                   │              │
        ◄── Verification ──────────┘──── Validation ──►
```

### Left Side (Verification — "Building the product right")

Each phase produces a specification that will be used to drive testing:
- **Requirements** → defines Acceptance Test criteria
- **System Design** → defines System Test criteria
- **Architectural Design** → defines Integration Test criteria
- **Module Design** → defines Unit Test criteria

### Right Side (Validation — "Building the right product")

Tests are designed *in parallel* with the corresponding left-side phase, even though they are executed later:
- **Unit Testing**: Verifies individual modules against module design
- **Integration Testing**: Verifies module interactions against architectural design
- **System Testing**: Verifies the full system against system design
- **Acceptance Testing**: Validates the system against user requirements

### Strengths Over Waterfall

- Testing is a first-class citizen, planned from the start
- Defects can be caught at a lower level (unit tests before system tests)
- Clear traceability from requirements to tests
- Suitable for safety-critical systems (aerospace, medical, defense)

### Weaknesses

- Still largely sequential; limited accommodation for changing requirements
- High documentation overhead
- Not suitable for exploratory or innovative projects

**Real-world example**: Developing embedded software for a medical infusion pump requires traceability from every requirement to a specific test, as mandated by FDA regulations. The V-Model's explicit verification/validation mapping satisfies regulatory requirements.

---

## 4. Incremental Model

The Incremental model delivers the system in a series of builds (increments), each adding functionality to the previous one. Requirements are typically defined upfront, but implementation is staged.

### Structure

```
Requirements ──► All planned upfront (or partially)
    │
    ├──► Increment 1: Core functionality
    │         Design ► Code ► Test ► Deploy
    │                                  │
    ├──► Increment 2: Additional features
    │         Design ► Code ► Test ► Deploy
    │                                  │
    ├──► Increment 3: Extended features
    │         Design ► Code ► Test ► Deploy
    │                                  │
    └──► Increment N: Final features
              Design ► Code ► Test ► Deploy
```

### Strengths

- Delivers working software earlier than Waterfall
- High-priority functionality is available sooner
- Users can provide feedback on early increments
- Risk is reduced: problems discovered early can be corrected

### Weaknesses

- Requires good architectural thinking upfront (each increment must fit into the overall design)
- Can lead to a "patched" architecture if not managed carefully
- Requirements for later increments may be poorly defined

---

## 5. Spiral Model

Barry Boehm introduced the Spiral model in 1988 in his paper "A Spiral Model of Software Development and Enhancement." The Spiral model is **risk-driven**: each cycle of the spiral focuses on identifying and mitigating the most significant risks before proceeding.

### Structure

The spiral consists of four quadrants, repeated in each cycle:

```
            Determine objectives,
            alternatives, constraints
                    │
                    ▼
    ┌───────────────────────────────┐
    │    Quadrant 1: PLANNING       │
    │  ┌──────────────────────────┐ │
    │  │       Quadrant 4:        │ │
    │  │    NEXT ITERATION        │◄├─── Accumulated risk ──►
    │  │       PLANNING           │ │                         │
    │  └──────────────────────────┘ │    Quadrant 2:         │
    └───────────────────────────────┘   RISK ANALYSIS        │
                    │                        │               │
                    ▼                        ▼               │
            Evaluate alternatives,    Identify, analyze,    │
            identify, resolve risks   resolve risks         │
                                           │               │
                    ┌──────────────────────┘               │
                    ▼                                       │
            Quadrant 3: DEVELOPMENT AND TESTING            │
            (Design, code, test the current increment)      │
                    │                                       │
                    └───────────────────────────────────────┘
```

More precisely, each spiral cycle contains:

1. **Determine objectives**: What does this cycle need to achieve? What are the constraints?
2. **Identify and resolve risks**: What could go wrong? Build prototypes to test uncertain areas.
3. **Development and validation**: Develop and test the product for this cycle.
4. **Plan the next cycle**: Review with stakeholders, decide whether to proceed.

### Risk-Driven Nature

The Spiral model is unique in placing risk analysis at the center of each iteration. If a risk cannot be resolved at reasonable cost, the project may be terminated early — this is a feature, not a bug, as it prevents investing further in a doomed project.

Example risks addressed in early spirals:
- "We are not sure users will adopt this interface" → Build a prototype and test with users
- "We don't know if this algorithm will be fast enough" → Build a performance spike
- "The vendor's API may not support our requirements" → Write an integration proof-of-concept

### Strengths

- Excellent risk management
- Flexible: can accommodate changing requirements between spirals
- Early prototyping reduces the chance of building the wrong product
- Suitable for large, complex, and high-risk projects

### Weaknesses

- Requires significant risk management expertise
- Can be expensive (each spiral cycle has overhead)
- Not suitable for small, low-risk projects
- End date can be unpredictable

**Real-world example**: A defense contractor developing a new command-and-control system faces significant technical risks (new communication protocols), organizational risks (changing requirements), and schedule risks. The Spiral model allows each risk to be addressed systematically before committing to full development.

---

## 6. RAD (Rapid Application Development)

Developed by James Martin in the early 1990s, RAD emphasizes extremely fast development through timeboxing, reuse, and heavy user involvement. RAD typically targets a 60–90 day delivery cycle.

### Key Principles

- **Timeboxing**: Fixed time periods (typically 60–90 days) regardless of how much remains
- **Prototyping**: Continuous prototype refinement with heavy user feedback
- **SWAT teams**: Small, highly skilled teams (Skilled With Advanced Tools)
- **Reuse**: Maximize use of existing components, frameworks, and tools
- **Continuous user involvement**: Users participate throughout development, not just at requirements and UAT

### RAD Phases

```
1. Requirements Planning ──► Identify business objectives and constraints
2. User Design           ──► Interactive prototyping with users (iterative)
3. Construction          ──► Rapid coding based on approved prototype
4. Cutover               ──► Testing, deployment, user training
```

### Strengths

- Fast delivery of working software
- High user satisfaction (heavy involvement)
- Reduced chance of building the wrong product

### Weaknesses

- Requires highly skilled, available users (stakeholder time is a major constraint)
- Not suitable for projects requiring new technology or complex algorithms
- Scalability challenges for large systems
- Can produce systems with poor performance or architecture if speed is prioritized over quality

---

## 7. Prototyping Model

The Prototyping model builds a working but incomplete version of the system to clarify requirements and test ideas before committing to full development.

### Types of Prototypes

**Throwaway (Exploratory) Prototyping**: Build a quick prototype to answer specific questions, then discard it. The prototype is not production code; it is a tool for learning.

**Evolutionary Prototyping**: Incrementally build the prototype into the final product. More risky because prototype code may carry quality problems into production.

### Process

```
┌─────────────────────────────────┐
│  1. Identify basic requirements  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  2. Build prototype             │◄─────────────┐
└────────────┬────────────────────┘             │
             │                                  │
             ▼                                  │
┌─────────────────────────────────┐             │
│  3. User evaluates prototype    │             │
└────────────┬────────────────────┘             │
             │                                  │
             ▼                                  │
       Acceptable? ──── No ──────────────────────┘
             │
            Yes
             │
             ▼
┌─────────────────────────────────┐
│  4. Build final system          │
└─────────────────────────────────┘
```

### When to Use Prototyping

- Requirements are unclear or evolving
- User interface design is a major concern
- Exploring feasibility of a new technology
- As part of a larger process model (e.g., prototyping during the risk resolution phase of a Spiral)

---

## 8. Comparison of SDLC Models

| Model | Requirements | Delivery | Risk | Documentation | Best For |
|-------|-------------|----------|------|---------------|----------|
| Waterfall | Must be stable upfront | Late (one release) | High if requirements wrong | Comprehensive | Stable, well-understood projects |
| V-Model | Must be stable upfront | Late (one release) | Medium (better testing) | Very comprehensive | Safety-critical systems |
| Incremental | Mostly upfront | Staged releases | Medium | Moderate | Priority-based delivery |
| Spiral | Evolving | Per-spiral releases | Low (risk-driven) | Moderate to heavy | Large, high-risk projects |
| RAD | Rapidly negotiated | Very fast | Low for simple projects | Light | Well-understood business domains |
| Prototyping | Unclear initially | After prototype approval | Low for requirements | Light | Unclear requirements, UI-heavy |
| Agile* | Evolving sprint-to-sprint | Continuous | Low | Lightweight | Dynamic requirements, collocated teams |

*Agile is covered in Lesson 3.

---

## 9. Choosing the Right Model

No model is universally best. The choice depends on multiple project factors:

### Decision Framework

```
Is the project safety-critical or highly regulated?
    └─ YES → V-Model or Waterfall with rigorous documentation
    └─ NO  ↓

Are requirements well-understood and stable?
    └─ YES → Waterfall or Incremental
    └─ NO  ↓

Are there significant technical or business risks?
    └─ YES → Spiral
    └─ NO  ↓

Is speed of delivery the primary concern?
    └─ YES → RAD (small team, business domain)
    └─ NO  ↓

Are requirements expected to change frequently?
    └─ YES → Agile (Scrum, Kanban — see Lesson 3)
    └─ NO → Incremental
```

### Key Decision Criteria

| Factor | Favors Sequential (Waterfall/V) | Favors Iterative/Agile |
|--------|--------------------------------|------------------------|
| Requirements stability | Stable, well-defined | Volatile, unclear |
| Team size | Large, distributed | Small to medium, co-located |
| Customer availability | Limited | High (frequent feedback) |
| Project duration | Short-medium | Medium-long |
| Technology novelty | Known technology | Cutting-edge or uncertain |
| Regulatory environment | High regulation | Low regulation |
| Contract type | Fixed-price, fixed-scope | Time-and-materials |

---

## 10. Real-World Examples

### Example 1: NASA Mission Software (V-Model)

NASA's flight software for crewed missions uses a strict V-Model process with formal verification at each phase. Requirements are traceable to tests; every test is reviewed by an independent team. The cost of a defect in space is potentially catastrophic, justifying the heavy process overhead.

### Example 2: E-Commerce Startup (Agile/Incremental)

A startup launching an online marketplace needs to reach market quickly, gather user feedback, and pivot based on what works. Waterfall would be fatal — by the time a full system is delivered 18 months later, the market may have changed. An agile approach with bi-weekly releases allows the team to learn and adapt continuously.

### Example 3: Banking Core System Replacement (Spiral)

A large bank replacing its 30-year-old core banking system faces enormous technical risks (mainframe migration), business risks (regulatory compliance), and organizational risks (training thousands of staff). The Spiral model allows the bank to prototype the most risky integrations early, validate them with regulators, and only proceed to full development once the critical risks are resolved.

### Example 4: Internal HR Tool (RAD)

A company's HR department needs a new tool for tracking employee certifications. Requirements are simple, the domain is well-understood, and a small team of two developers can deliver in six weeks with heavy HR manager involvement. RAD (or a simple iterative approach) is appropriate.

---

## Summary

The Software Development Life Cycle provides the structural framework within which software is built. The choice of SDLC model shapes every aspect of the project: how teams are organized, when deliverables are produced, how risk is managed, and how changes are handled.

Key takeaways:
- **Waterfall** is sequential and documentation-heavy; appropriate for stable, well-understood requirements
- **V-Model** extends Waterfall with explicit test planning at each phase; preferred for safety-critical systems
- **Incremental** delivers software in stages; balances structure with earlier delivery
- **Spiral** is risk-driven; each iteration resolves the most significant risks first
- **RAD** emphasizes extreme speed through timeboxing and user involvement
- **Prototyping** builds incomplete systems to clarify requirements before full development
- No single model is universally best; the choice depends on requirements stability, risk, team size, and regulatory context

---

## Practice Exercises

**Exercise 1**: A city government is building a new traffic management system that will control 800 intersections citywide. The project has a fixed budget, regulatory requirements for safety certification, and three years to complete. Which SDLC model would you recommend? Justify your choice by addressing at least three of the decision criteria from Section 9.

**Exercise 2**: Draw a Waterfall timeline for the following project: A company needs a mobile app that allows employees to submit expense reports. The app must integrate with the company's existing SAP ERP system. Estimate the duration of each phase (as percentages of total project time) and list two key artifacts produced in each phase.

**Exercise 3**: A new social networking startup wants to build a platform but has only a rough idea of what features users will want. Why would Waterfall be a poor choice? Design a three-sprint incremental plan for their first six weeks, specifying what features would be included in each increment.

**Exercise 4**: Compare the V-Model and Waterfall models. Create a table listing five specific differences. Under what circumstances would you choose V-Model over Waterfall despite its higher documentation overhead?

**Exercise 5**: The Spiral model uses prototypes to resolve risks. For a project building a real-time fraud detection system for a bank, identify three significant risks and describe what prototype or spike you would build in an early spiral to resolve each risk.

---

## Further Reading

- **Winston W. Royce**, "Managing the Development of Large Software Systems" (1970) — The paper that introduced what became known as Waterfall
- **Barry W. Boehm**, "A Spiral Model of Software Development and Enhancement" (1988) — IEEE Computer, Vol. 21, No. 5
- **James Martin**, *Rapid Application Development* (1991) — Macmillan
- **Ian Sommerville**, *Software Engineering* (10th ed.) — Chapters 2 and 3 cover process models in depth
- **Roger Pressman**, *Software Engineering: A Practitioner's Approach* — Chapter 2: Process Models
- **SEI CMMI**: https://cmmiinstitute.com — Capability Maturity Model Integration, which spans all process models

---

**Previous**: [What Is Software Engineering](./01_What_Is_Software_Engineering.md) | **Next**: [Agile and Iterative Development](./03_Agile_and_Iterative_Development.md)
