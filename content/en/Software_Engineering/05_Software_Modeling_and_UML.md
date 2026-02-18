# Lesson 05: Software Modeling and UML

**Previous**: [04. Requirements Engineering](./04_Requirements_Engineering.md) | **Next**: [06. Estimation and Planning](./06_Estimation_and_Planning.md)

---

Before a single line of code is written, engineers think in terms of models. A model is an abstract, simplified representation of a system that highlights some aspects while hiding others. Good models sharpen communication, expose design problems early, and serve as documentation long after memories have faded. The **Unified Modeling Language (UML)** is the industry-standard notation for software models. This lesson teaches you the most useful UML diagrams, when to apply each one, and how to read and draw them correctly.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Lesson 01 — What Is Software Engineering
- Lesson 04 — Requirements Engineering (use cases will be revisited here)
- Basic object-oriented programming concepts (class, object, inheritance)

**Learning Objectives**:
- Explain why software models are valuable and at what level of detail to draw them
- Distinguish structural diagrams from behavioral diagrams in UML 2.x
- Draw and interpret use case diagrams with actors, use cases, and relationships
- Draw class diagrams showing classes, attributes, operations, and all five relationship types
- Draw sequence diagrams with lifelines, messages, activation boxes, and combined fragments
- Draw activity diagrams with decision nodes, forks/joins, and swimlanes
- Draw state machine diagrams with states, transitions, guards, and entry/exit actions
- Describe the purpose of component and deployment diagrams
- Apply modeling best practices and avoid common mistakes

---

## Table of Contents

1. [Why Model Software?](#1-why-model-software)
2. [UML Overview](#2-uml-overview)
3. [Use Case Diagrams](#3-use-case-diagrams)
4. [Class Diagrams](#4-class-diagrams)
5. [Sequence Diagrams](#5-sequence-diagrams)
6. [Activity Diagrams](#6-activity-diagrams)
7. [State Machine Diagrams](#7-state-machine-diagrams)
8. [Component and Deployment Diagrams](#8-component-and-deployment-diagrams)
9. [When to Use Which Diagram](#9-when-to-use-which-diagram)
10. [Modeling Best Practices and Common Mistakes](#10-modeling-best-practices-and-common-mistakes)
11. [Summary](#11-summary)
12. [Practice Exercises](#12-practice-exercises)
13. [Further Reading](#13-further-reading)

---

## 1. Why Model Software?

### 1.1 The Role of Abstraction

A map of a city does not include every crack in the pavement — it shows the roads and landmarks that matter for navigation. Software models work the same way: they suppress irrelevant detail to make the important structure visible.

Benefits of modeling:
- **Communication**: a diagram conveys structure to stakeholders, teammates, and future maintainers faster than prose or code
- **Early defect detection**: design flaws surfaced in a model cost far less to fix than defects found in code or production
- **Blueprint for construction**: models guide implementation choices
- **Documentation**: living models kept in sync with the system serve as up-to-date architecture documentation

### 1.2 The Right Level of Abstraction

The right level depends on the audience and purpose:

| Audience | Appropriate model | Level |
|---|---|---|
| Business stakeholders | Use case diagram, activity diagram | Business process |
| Architects | Component / deployment diagram | System / subsystem |
| Developers | Class diagram, sequence diagram | Module / class |
| Testers | State machine, activity diagram | Behavior / edge cases |

Over-modeling wastes time and produces diagrams no one reads. Under-modeling leaves teams with a shared misunderstanding. A rule of thumb: model the parts that are novel, complex, or likely to cause misunderstanding.

### 1.3 Models as Executable Specifications

Some teams use model-driven development (MDD) tools (e.g., Enterprise Architect, Modelio) to generate code skeletons from class and state machine diagrams. In safety-critical domains (automotive AUTOSAR, avionics DO-178C), models can be the authoritative artifact.

---

## 2. UML Overview

### 2.1 History

UML was created by Grady Booch, Ivar Jacobson, and James Rumbaugh (the "Three Amigos") at Rational Software in the mid-1990s, unifying several competing OO notations. Version 1.0 was submitted to the Object Management Group (OMG) in 1997. UML 2.0 (2004) significantly restructured the language. The current standard is **UML 2.5.1 (2017)**.

### 2.2 Diagram Taxonomy

UML 2.x defines 14 diagram types in two families:

**Structural Diagrams** — describe the static structure (what exists):

| Diagram | Describes |
|---|---|
| Class | Classes, attributes, operations, relationships |
| Object | Instance-level snapshot of a class diagram |
| Component | Software components and their interfaces |
| Composite Structure | Internal structure of a class or component |
| Package | Packages and their dependencies |
| Deployment | Hardware nodes and artifact deployment |
| Profile | UML extension mechanisms |

**Behavioral Diagrams** — describe dynamic behavior (what happens):

| Diagram | Describes |
|---|---|
| Use Case | System functions from a user's perspective |
| Activity | Workflow; parallel and sequential actions |
| State Machine | States of an object and transitions between them |
| Sequence | Object interactions ordered by time |
| Communication | Object interactions emphasizing links |
| Timing | State changes against a time axis |
| Interaction Overview | High-level flow of interaction fragments |

In practice, the five most commonly used diagrams are: **use case, class, sequence, activity, and state machine**. This lesson covers those five in depth.

---

## 3. Use Case Diagrams

A use case diagram shows who interacts with the system (actors) and what they can do (use cases). It captures **functional requirements** at a high level of abstraction.

### 3.1 Elements

| Element | Notation | Description |
|---|---|---|
| Actor | Stick figure | External entity (person, system, device) that interacts with the system |
| Use Case | Oval | A unit of system behavior visible to actors |
| System Boundary | Rectangle | Separates internal use cases from external actors |
| Association | Solid line | Actor participates in the use case |
| Include | Dashed arrow + `<<include>>` | A use case always includes another (shared sub-behavior) |
| Extend | Dashed arrow + `<<extend>>` | A use case optionally extends another (conditional behavior) |
| Generalization | Solid arrow (hollow head) | One actor/use case specializes another |

### 3.2 Include vs. Extend

- **`<<include>>`**: the base use case *always* calls the included use case. Use it to extract reusable behavior. Arrow points *from* base *to* included.
- **`<<extend>>`**: the extending use case *may* add behavior to the base case under certain conditions. Arrow points *from* extension *to* base.

Memory aid: `<<include>>` is like a function call (always happens); `<<extend>>` is like a plugin (sometimes happens).

### 3.3 ASCII Art Example: Online Library System

```
            ┌─────────────────────────────────────────┐
            │           Online Library System          │
            │                                          │
  Member ───┼──── Search Catalog                       │
    │        │         │ <<include>>                    │
    │        │    View Book Details ◄── Extend ── Reserve Book
    │        │                                   (if logged in)
    │        │                                          │
    ├────────┼──── Borrow Book ─────<<include>>──────► Authenticate
    │        │                                          │
    │        │                                          │
  Librarian ─┼──── Manage Catalog                       │
    │        │                                          │
    │        │                                          │
   Admin ────┼──── Manage Users                         │
            │                                          │
            └─────────────────────────────────────────┘
```

*(In a formal tool, arrows would be correctly oriented. ASCII art approximates the structure.)*

### 3.4 Use Case Descriptions

A use case diagram alone is not sufficient. Each use case should have a **prose description** covering:

```
Use Case: Borrow Book
Actor: Member (primary), Librarian (secondary)
Preconditions: Member is authenticated. Book is available.
Main Success Scenario:
  1. Member scans book barcode.
  2. System verifies the book is available for loan.
  3. System records loan: member ID, book ID, due date (today + 14 days).
  4. System decrements available copy count.
  5. System prints/displays a loan receipt.
Postconditions: Loan record exists. Book copy count decreased by 1.
Alternative Flows:
  2a. Book is on hold for another member.
     System notifies librarian; loan is refused.
  2b. Member has exceeded the loan limit (5 books).
     System displays error; loan is refused.
```

### 3.5 Common Mistakes

- **Too many use cases**: keep them at the level of a complete user goal, not individual system steps
- **Actor is a job title, not a role**: a "Sales Manager" should be modeled as "Approver" or "Reporter" if that is the relevant role
- **Overusing `<<extend>>`**: most optional behavior is better captured in the flow narrative
- **Putting business logic inside use case names**: "Calculate Monthly Discount If Premium Member" is a step, not a use case

---

## 4. Class Diagrams

The class diagram is the most fundamental UML diagram for object-oriented design. It shows the static structure of the system: the classes (or types), their attributes, their operations, and the relationships between them.

### 4.1 Class Notation

```
┌──────────────────────────────┐
│        BankAccount           │  ← Class name (centered, bold)
├──────────────────────────────┤
│ - accountNumber: String      │  ← Attributes
│ - balance: Decimal           │    visibility: - private, + public,
│ # owner: Customer            │    # protected, ~ package
│ + interestRate: float        │
├──────────────────────────────┤
│ + deposit(amount: Decimal)   │  ← Operations (methods)
│ + withdraw(amount: Decimal)  │    return types follow colon
│ - calculateInterest(): float │
│ + getBalance(): Decimal      │
└──────────────────────────────┘
```

### 4.2 Relationship Types

| Relationship | Notation | Meaning | Strength |
|---|---|---|---|
| **Dependency** | Dashed arrow | A uses B (transient) | Weakest |
| **Association** | Solid line | A has a reference to B | — |
| **Aggregation** | Solid line + hollow diamond at A | A owns B but B can exist without A | — |
| **Composition** | Solid line + filled diamond at A | A owns B; B cannot exist without A | Stronger |
| **Inheritance** | Solid line + hollow triangle at parent | A is a B (generalization) | — |
| **Realization** | Dashed line + hollow triangle | A implements interface B | — |

### 4.3 Multiplicity

Multiplicity appears at both ends of an association line:

| Notation | Meaning |
|---|---|
| `1` | Exactly one |
| `0..1` | Zero or one (optional) |
| `*` or `0..*` | Zero or more |
| `1..*` | One or more |
| `2..5` | Between two and five |

### 4.4 ASCII Art Example: E-Commerce Order System

```
Customer                 Order                    Product
┌──────────────┐  1    *┌──────────────┐  *    *┌──────────────┐
│- customerId  ├────────┤- orderId     ├─────────┤- productId   │
│- name        │        │- orderDate   │         │- name        │
│- email       │        │- status      │         │- price       │
├──────────────┤        ├──────────────┤         │- stockQty    │
│+ register()  │        │+ addItem()   │         ├──────────────┤
│+ login()     │        │+ cancel()    │         │+ checkStock()│
│+ getOrders() │        │+ getTotal()  │         │+ reserve()   │
└──────────────┘        └──────┬───────┘         └──────────────┘
                               │ 1
                               │  (composition: line items
                               │   cannot exist without order)
                              ◆│
                            * │
                    ┌──────────▼──────┐
                    │   OrderItem     │
                    │- quantity: int  │
                    │- unitPrice      │
                    ├─────────────────┤
                    │+ getSubtotal()  │
                    └─────────────────┘

PaymentMethod (abstract)
┌──────────────────┐
│ <<abstract>>     │
│+ authorize()     │
│+ capture()       │
└────────┬─────────┘
         │ (inheritance)
    ┌────┴──────────────┐
    │                   │
CreditCard           BankTransfer
┌──────────┐       ┌──────────────┐
│- cardNum │       │- accountNum  │
│- expiry  │       │- routingNum  │
└──────────┘       └──────────────┘
```

### 4.5 Interfaces and Abstract Classes

- An **interface** (or `<<interface>>`) declares operations without implementation. Classes *realize* interfaces.
- An **abstract class** (name in *italics* in formal UML, or `{abstract}` tag) cannot be instantiated; subclasses must implement abstract operations.

### 4.6 Common Mistakes

- **God class**: one class with 20+ attributes and 30+ operations. Break it up by single responsibility.
- **Missing multiplicity**: omitting `1` or `*` leaves ambiguity about cardinality.
- **Confusing aggregation and composition**: use composition only when the child's lifecycle is strictly dependent on the parent's.
- **Relationship overuse**: not every association needs a named role. Reserve role names for clarity.
- **Putting code in diagrams**: operations in class diagrams show signatures, not implementations.

---

## 5. Sequence Diagrams

A sequence diagram shows how objects interact over time to carry out a scenario — typically one main flow of a use case. Time flows top-to-bottom; the x-axis is object identity.

### 5.1 Elements

| Element | Notation | Description |
|---|---|---|
| **Lifeline** | Vertical dashed line | Represents one participant (object, component, actor) |
| **Activation box** | Thin rectangle on lifeline | The period during which the object is executing |
| **Synchronous message** | Solid arrowhead `→` | Caller waits for return |
| **Return message** | Dashed arrow `-->` | Return value from called object |
| **Asynchronous message** | Open arrowhead `->` | Caller does not wait |
| **Self-message** | Arrow looping back to same lifeline | Object calls itself (recursion) |
| **Create message** | Dashed arrow to box head | Instantiates a new object |
| **Destroy** | X at bottom of lifeline | Object is deleted |

### 5.2 Combined Fragments

Combined fragments add control flow notation:

| Operator | Meaning |
|---|---|
| `alt` | Alternative (if/else); each branch separated by dashed line |
| `opt` | Optional; executed only if guard is true |
| `loop` | Repeated; `loop(min, max)` or `loop(condition)` |
| `par` | Parallel execution of sub-fragments |
| `break` | Abort the enclosing fragment |
| `ref` | Reference to another interaction diagram |
| `critical` | Atomic region; not interleaved |

### 5.3 ASCII Art Example: User Login Sequence

```
  :User       :Browser      :AuthController  :UserRepository  :SessionStore
    |              |                |                 |               |
    |--submit()-->  |                |                 |               |
    |              |--POST /login-->|                 |               |
    |              |                |--findByEmail()-->|               |
    |              |                |<--User or null--|               |
    |              |  ┌─alt──────────────────────────────────────────┐|
    |              |  │ [user found AND password matches]             ||
    |              |  │              |--createSession(userId)-------->||
    |              |  │              |<----sessionToken---------------|
    |              |  │              |                                ||
    |              |<-│-200 OK + token                               ||
    |<--redirect-->|  │                                              ||
    |              |  ├──────────────────────────────────────────────┤|
    |              |  │ [user not found OR password wrong]            ||
    |              |<-│-401 Unauthorized                             ||
    |<--show error-|  └──────────────────────────────────────────────┘|
    |              |                |                 |               |
```

### 5.4 Sequence vs. Communication Diagrams

Both show object interactions. Sequence diagrams emphasize **time ordering** (left-to-right or top-to-bottom); communication (collaboration) diagrams emphasize **link structure** between objects. Sequence diagrams are more widely used for documenting use case scenarios.

### 5.5 Common Mistakes

- **Too many lifelines**: keep to 4–7 participants per diagram; use `ref` fragments to break long scenarios
- **Missing return arrows**: omitting returns makes it ambiguous whether a call is synchronous
- **Including all system internals**: show the objects and messages relevant to the scenario, not every internal call
- **Flat diagrams without combined fragments**: add `alt`, `loop`, and `opt` to show non-trivial control flow

---

## 6. Activity Diagrams

An activity diagram models a **workflow** or **algorithm** — the flow of control and data through a sequence of actions. It is UML's equivalent of a flowchart, but with additional features for concurrent execution.

### 6.1 Elements

| Element | Notation | Description |
|---|---|---|
| **Initial node** | Filled circle | Starting point |
| **Activity final node** | Filled circle inside ring | End of entire activity |
| **Flow final node** | X inside circle | Ends one branch, not the whole activity |
| **Action** | Rounded rectangle | A single step or task |
| **Decision/Merge node** | Diamond | Branch (decision) or join branches (merge) |
| **Fork** | Thick horizontal bar | Splits flow into parallel branches |
| **Join** | Thick horizontal bar | Synchronizes parallel branches |
| **Swimlane (partition)** | Vertical/horizontal lane | Assigns responsibility to a role or component |
| **Object node** | Rectangle | Data produced or consumed |
| **Control flow** | Arrow | Sequence between actions |

### 6.2 ASCII Art Example: Order Processing (with swimlanes)

```
  Customer              Website               Warehouse            Finance
     │                     │                      │                   │
  ●  │                     │                      │                   │
     │──Place Order──────►  │                      │                   │
     │                     │──Validate Payment──────────────────────► │
     │                     │◄───────────────────────────────OK────── │
     │                     │                      │                   │
     │                     │──Send Order────────► │                   │
     │                     │                     ═══ (Fork)           │
     │                     │              Pick Items│   │Update Stock  │
     │                     │                      │   │               │
     │                     │                     ═══ (Join)           │
     │                     │◄──Shipment Confirmed─ │                   │
     │◄──Confirmation Email─ │                      │                   │
  ⊙  │                     │                      │                   │
```

### 6.3 When to Use Activity Diagrams

- Documenting **business processes** (order to cash, employee onboarding)
- Modeling **algorithmic workflows** with branching and looping
- Showing **concurrent processes** (fork/join)
- Capturing **cross-functional responsibilities** (via swimlanes)

Activity diagrams are preferred over sequence diagrams when the focus is *flow of control* rather than *object interactions*.

### 6.4 Common Mistakes

- **Missing merge before decision**: a decision node that has two incoming flows (from both branches) should have an explicit merge node before it
- **Over-complex diagrams**: if an activity diagram exceeds one page, use sub-activities (call behavior actions) to decompose it
- **Confusing fork with decision**: fork creates parallel flows; decision creates exclusive alternative flows

---

## 7. State Machine Diagrams

A state machine diagram models the **discrete states** of an object and the **transitions** between those states in response to events. State machines are especially useful for objects whose behavior depends heavily on their history.

### 7.1 Elements

| Element | Notation | Description |
|---|---|---|
| **State** | Rounded rectangle | A condition the object can be in |
| **Initial pseudostate** | Filled circle | Entry point |
| **Final state** | Filled circle in ring | Terminal state |
| **Transition** | Arrow between states | Triggered change |
| **Guard** | `[condition]` on transition | Boolean condition that enables the transition |
| **Trigger** | Event name before `/` | The event that fires the transition |
| **Action** | After `/` | Behavior executed on the transition |
| **Entry action** | `entry / action` in state | Executed on entering the state |
| **Exit action** | `exit / action` in state | Executed on leaving the state |
| **Do activity** | `do / activity` in state | Ongoing activity while in the state |

### 7.2 Transition Syntax

```
trigger [guard] / action
```

Example: `paymentReceived [amount >= total] / confirmOrder()`

### 7.3 ASCII Art Example: Order Lifecycle

```
     ●
     │
     ▼
┌──────────┐   itemsAdded       ┌──────────────┐
│  DRAFT   │──────────────────► │  PENDING     │
│          │                    │  PAYMENT     │
└──────────┘                    └──────┬───────┘
                                       │
              paymentReceived [valid]  │    paymentFailed
              ◄─────────────────────── ┤ ──────────────────────►
              │                        │                       ┌──────────┐
              ▼                        │                       │ PAYMENT  │
      ┌───────────────┐                │                       │ FAILED   │
      │   CONFIRMED   │                │                       └──────────┘
      │ entry/notifyWH│                │
      │ do/reserveItems│               │
      └───────┬───────┘               │
              │ shipped               │
              ▼                       │
      ┌───────────────┐               │
      │   SHIPPED     │               │
      │ entry/sendEmail│              │
      └───────┬───────┘               │
              │ delivered             │
              ▼                       │
      ┌───────────────┐               │
      │   DELIVERED   │               │
      └───────┬───────┘               │
              │                       │
        ┌─────┴─────┐  customerReturn │
        │           │────────────────►┌──────────┐
        │           │                 │ RETURNED │
        │           │                 └──────────┘
        ▼           │
       ⊙            │
```

### 7.4 Composite States and History

A **composite state** (state with internal states) allows hierarchical state machines. A **history pseudostate** (`H`) remembers which sub-state was active when the composite was exited, resuming there on re-entry.

### 7.5 Common Mistakes

- **Missing initial transition**: every state machine must have an initial pseudostate
- **Unguarded competing transitions**: if two transitions from the same state have the same trigger and no guards, behavior is non-deterministic
- **Modeling too many objects**: state machines are for individual objects, not entire systems
- **Confusing state with value**: a state should capture a behavioral mode (how the object responds), not just a data value

---

## 8. Component and Deployment Diagrams

### 8.1 Component Diagrams

A component diagram shows the **software components** (modules, services, executables, libraries) that make up the system and their **interfaces** and **dependencies**.

Key notation:
- **Component**: box with `<<component>>` stereotype or component icon (box with two small boxes on left edge)
- **Interface provided**: "lollipop" (circle on a line)
- **Interface required**: "socket" (half-circle on a line)
- **Dependency**: dashed arrow

```
                   ┌──────────────────┐
                   │  <<component>>   │
                   │   OrderService   ○── IOrderPort
                   │                  │
                   └────────┬─────────┘
                            │ uses
                     ┌──────┴────────────────┐
             ┌───────▼──────┐  ┌─────────────▼─────┐
             │ <<component>>│  │  <<component>>     │
             │ PaymentGW    │  │  InventoryService  │
             └──────────────┘  └────────────────────┘
```

### 8.2 Deployment Diagrams

A deployment diagram shows how software components are mapped to physical or virtual **nodes** (hardware, VMs, containers, cloud services).

Key notation:
- **Node**: 3D box (hardware) or `<<device>>`, `<<executionEnvironment>>`
- **Artifact**: deployed executable or file (`<<artifact>>`)
- **Communication path**: solid line between nodes

```
┌────────────────────────┐       ┌────────────────────────┐
│ <<device>>             │       │ <<device>>             │
│  Web Server (AWS EC2)  │───────│  DB Server (RDS)       │
│  ┌──────────────────┐  │  TCP  │  ┌──────────────────┐  │
│  │ <<artifact>>     │  │ 5432  │  │ <<artifact>>     │  │
│  │  app.war         │  │       │  │  postgres DB     │  │
│  └──────────────────┘  │       │  └──────────────────┘  │
└────────────────────────┘       └────────────────────────┘
            │
         HTTPS
            │
┌───────────▼────────────┐
│ <<device>>             │
│  Client Browser        │
└────────────────────────┘
```

Deployment diagrams are essential for:
- Communicating infrastructure architecture to DevOps and operations teams
- Planning network security zones
- Container and microservices architecture documentation

---

## 9. When to Use Which Diagram

| Goal | Best Diagram |
|---|---|
| Communicate system scope to stakeholders | Use Case |
| Document detailed use case flow | Sequence |
| Design object structure and relationships | Class |
| Show a workflow or business process | Activity |
| Model object lifecycle (e.g., order states) | State Machine |
| Document system components and APIs | Component |
| Show infrastructure and deployment | Deployment |
| Explore parallel and concurrent flows | Activity (fork/join) |
| Show data transformation through pipeline | Activity (object nodes) |
| Document algorithmic branching | Activity or Sequence (with alt/opt) |

A practical heuristic:
- **Start** with a use case diagram to define scope
- **Elaborate** key scenarios with sequence diagrams
- **Design** the domain model with a class diagram
- **Add** state machines for lifecycle-heavy objects
- **Document** architecture with component and deployment diagrams

---

## 10. Modeling Best Practices and Common Mistakes

### 10.1 Best Practices

1. **Model with a purpose**: every diagram should have a stated audience and a question it answers. If you cannot articulate why a diagram exists, do not draw it.

2. **Prefer sketches over perfection**: a whiteboard photo of a rough diagram that gets the idea across is more valuable than a pixel-perfect diagram that took three days and is already out of date.

3. **Keep diagrams small**: a diagram that requires scrolling or zooming will not be read. Split large diagrams using packages, sub-activities, or `ref` frames.

4. **Maintain consistency**: if a class is called `OrderItem` in the class diagram, it should be called `OrderItem` everywhere — not `LineItem` in the sequence diagram and `Item` in the database schema.

5. **Version-control your models**: store model source files (`.xmi`, `.puml`, `.drawio`) in the repository alongside code. Generate diagram images in CI.

6. **Use PlantUML or Mermaid for text-based models**: they integrate with code review tools and prevent diagrams from drifting out of sync with code.

7. **Validate with stakeholders**: every diagram should be walked through with at least one non-author before being treated as authoritative.

### 10.2 Common Mistakes Summary

| Mistake | Impact | Fix |
|---|---|---|
| Modeling everything | Time waste; diagrams nobody reads | Model only what is novel or contentious |
| Wrong diagram type | Miscommunication | Match diagram to question (see §9) |
| Inconsistent naming | Confusion, implementation errors | Agree on a glossary; apply it everywhere |
| Stale diagrams | False confidence in documentation | Update diagrams in same PR as code |
| Missing multiplicity | Ambiguous cardinality leads to wrong schema | Always annotate both ends of associations |
| Overused `<<extend>>` | Complex, hard-to-read use case diagrams | Prefer narrative descriptions for optional flows |
| Ignoring NFRs | Diagrams capture structure but miss performance, security | Annotate constraints on component/deployment diagrams |

### 10.3 Tools

| Tool | Type | Notes |
|---|---|---|
| **PlantUML** | Text-based, open source | Integrates with IDEs, CI; good for version control |
| **Mermaid** | Text-based, JS-rendered | GitHub/GitLab native; excellent for sequence and class |
| **draw.io / diagrams.net** | GUI, free | Good for ad-hoc diagrams; XML format is version-controllable |
| **Lucidchart** | GUI, cloud | Collaboration-friendly; Jira/Confluence integration |
| **Enterprise Architect** | Full CASE tool | For large teams requiring MDD and code generation |
| **Visual Paradigm** | Full CASE tool | UML + BPMN + ArchiMate support |

---

## 11. Summary

UML provides a standardized vocabulary for modeling software at multiple levels of abstraction. The five most important diagrams are:

| Diagram | Family | Core question answered |
|---|---|---|
| Use Case | Behavioral | What can users do with the system? |
| Class | Structural | What types exist and how are they related? |
| Sequence | Behavioral | Which objects collaborate in what order? |
| Activity | Behavioral | What is the flow of this process or algorithm? |
| State Machine | Behavioral | How does this object's behavior depend on its state? |

Key principles:
- Model **purposefully**: choose the diagram that answers the question at hand
- Keep diagrams **small and focused**; use references for large systems
- Maintain **consistency** with code and other artifacts
- Use **text-based tools** (PlantUML, Mermaid) for diagrams that live alongside code
- Always **validate** models with the intended audience

UML is a means to an end — better software. A perfect diagram that does not improve shared understanding or catch a design flaw has failed its purpose.

---

## 12. Practice Exercises

**Exercise 1: Use Case Diagram**

Draw a use case diagram for a hotel booking system. The system supports three actors: Guest, Receptionist, and System Administrator. Include at least six use cases and at least one `<<include>>` and one `<<extend>>` relationship. Write a brief use case description (preconditions, main flow, alternative flow) for the "Book Room" use case.

---

**Exercise 2: Class Diagram**

Model a university course registration system with the following classes: Student, Course, Enrollment, Instructor, Department. Include:
- At least one inheritance hierarchy (e.g., different student types)
- Correct multiplicity on all associations
- At least one composition and one aggregation
- At least five attributes and three operations per class

---

**Exercise 3: Sequence Diagram**

Draw a sequence diagram for the scenario "Student enrolls in a course" from your Exercise 2 system. The scenario should involve at least four objects and include at least one `alt` fragment (e.g., course is full vs. not full) and one `loop` or `opt` fragment.

---

**Exercise 4: Activity and State Machine**

For an ATM system:

a. Draw an **activity diagram** for the "Withdraw Cash" use case, including the steps: insert card, validate PIN (with retry logic, max 3 attempts), select amount, check balance, dispense cash, print receipt, eject card.

b. Draw a **state machine diagram** for the ATM itself, with states: Idle, CardInserted, PINEntry, Authenticated, TransactionInProgress, Dispensing, OutOfService.

---

**Exercise 5: Architecture Diagrams**

You are designing a microservices e-commerce backend. Draw:

a. A **component diagram** showing at least five services (e.g., API Gateway, Order Service, Inventory Service, Payment Service, Notification Service) with their required and provided interfaces.

b. A **deployment diagram** showing these services deployed on Kubernetes pods, with a load balancer, a PostgreSQL database cluster (primary + replica), and a Redis cache.

---

## 13. Further Reading

- Rumbaugh, J., Jacobson, I., Booch, G. — *The Unified Modeling Language Reference Manual* (2nd ed., Addison-Wesley, 2004) — the authoritative reference by UML's creators
- Fowler, M. — *UML Distilled* (3rd ed., Addison-Wesley, 2003) — short, practical guide; best first book on UML
- OMG UML Specification 2.5.1 — https://www.omg.org/spec/UML/ — the normative specification
- PlantUML documentation — https://plantuml.com/ — text-based UML with IDE integration
- Mermaid documentation — https://mermaid.js.org/ — browser-native diagramming for GitHub/GitLab
- Scott, K. — *Fast Track UML 2.0* (Apress, 2004) — concise reference with worked examples
- Evans, E. — *Domain-Driven Design* (Addison-Wesley, 2003) — how class models reflect domain concepts

---

**Previous**: [04. Requirements Engineering](./04_Requirements_Engineering.md) | **Next**: [06. Estimation and Planning](./06_Estimation_and_Planning.md)
