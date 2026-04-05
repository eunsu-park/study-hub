# Microservices Fundamentals

**Previous**: [Message Systems Comparison](./12_Message_System_Comparison.md) | **Next**: [Microservices Patterns](./14_Microservices_Patterns.md)

Difficulty: ⭐⭐⭐

## Learning Objectives

1. Compare monolithic and microservices architectures, identifying the trade-offs of each approach
2. Enumerate the key characteristics of well-designed microservices (single responsibility, autonomous deployment, decentralized data)
3. Apply Domain-Driven Design concepts (bounded contexts, aggregates) to define service boundaries
4. Explain the database-per-service pattern and strategies for managing cross-service data consistency
5. Distinguish between synchronous (REST, gRPC) and asynchronous (event-driven) inter-service communication styles
6. Outline a practical migration strategy from a monolith to microservices using the Strangler Fig pattern

---

Netflix famously migrated from a monolithic Java application to over 700 microservices, enabling them to deploy thousands of times per day and scale individual components independently. Yet many teams that rush into microservices without understanding the fundamentals end up with a "distributed monolith" that is harder to operate than what they started with. This lesson builds the conceptual foundation you need to decompose systems wisely and avoid the most common pitfalls.

## Table of Contents

1. [Monolith vs Microservices](#1-monolith-vs-microservices)
2. [Characteristics of Microservices](#2-characteristics-of-microservices)
3. [Defining Service Boundaries](#3-defining-service-boundaries)
4. [Database per Service](#4-database-per-service)
5. [Inter-Service Communication](#5-inter-service-communication)
6. [Microservices Adoption Strategy](#6-microservices-adoption-strategy)
7. [Practice Problems](#7-practice-problems)

---

## 1. Monolith vs Microservices

### Monolithic Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Monolithic Architecture                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Monolithic Application                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │ │
│  │  │    User     │  │   Order     │  │   Product   │               │ │
│  │  │   Module    │  │   Module    │  │   Module    │               │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │ │
│  │         │                │                │                       │ │
│  │         └────────────────┼────────────────┘                       │ │
│  │                          │                                        │ │
│  │                          ▼                                        │ │
│  │              ┌─────────────────────┐                             │ │
│  │              │    Shared Library   │                             │ │
│  │              │   (Common Utils)    │                             │ │
│  │              └──────────┬──────────┘                             │ │
│  │                         │                                         │ │
│  │                         ▼                                         │ │
│  │              ┌─────────────────────┐                             │ │
│  │              │   Single Database   │                             │ │
│  │              └─────────────────────┘                             │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Characteristics:                                                       │
│  - Single codebase, single deployment unit                              │
│  - Direct function calls between modules                                │
│  - One shared database                                                  │
│  - Same technology stack                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Microservices Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Microservices Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │
│  │  User Service │  │ Order Service │  │Product Service│              │
│  │  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │              │
│  │  │  API    │  │  │  │  API    │  │  │  │  API    │  │              │
│  │  └─────────┘  │  │  └─────────┘  │  │  └─────────┘  │              │
│  │  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │              │
│  │  │ Logic   │  │  │  │ Logic   │  │  │  │ Logic   │  │              │
│  │  └─────────┘  │  │  └─────────┘  │  │  └─────────┘  │              │
│  │       │       │  │       │       │  │       │       │              │
│  │       ▼       │  │       ▼       │  │       ▼       │              │
│  │  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │              │
│  │  │   DB    │  │  │  │   DB    │  │  │  │   DB    │  │              │
│  │  │(Postgres│  │  │  │ (MySQL) │  │  │  │ (Mongo) │  │              │
│  │  └─────────┘  │  │  └─────────┘  │  │  └─────────┘  │              │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘              │
│          │                  │                  │                       │
│          └──────────────────┼──────────────────┘                       │
│                             │                                          │
│                      Network Communication                             │
│                     (REST/gRPC/MQ)                                     │
│                                                                         │
│  Characteristics:                                                       │
│  - Independent deployment units                                         │
│  - Network communication between services                               │
│  - Independent database per service                                     │
│  - Various technology stacks possible                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pros and Cons Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Monolith vs Microservices                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┬───────────────────┬───────────────────┐            │
│  │     Aspect     │     Monolith      │   Microservices   │            │
│  ├────────────────┼───────────────────┼───────────────────┤            │
│  │ Early Dev      │ Fast              │ Slow              │            │
│  │ Deployment     │ Full redeploy     │ Independent       │            │
│  │ Scaling        │ Scale all         │ Selective scaling │            │
│  │ Failure Impact │ Entire system     │ Affected service  │            │
│  │ Tech Stack     │ Single            │ Various           │            │
│  │ Team Structure │ Single team       │ Team per service  │            │
│  │ Transactions   │ ACID possible     │ Distributed txn   │            │
│  │ Testing        │ Simple            │ Complex           │            │
│  │ Debugging      │ Easy              │ Difficult         │            │
│  │ Ops Complexity │ Low               │ High              │            │
│  │ Network        │ None              │ Required (latency,│            │
│  │                │                   │ failures)         │            │
│  └────────────────┴───────────────────┴───────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Problems with Monolith

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Common Monolith Problems                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Big Ball of Mud                                                     │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                                                               │     │
│  │    ┌───┐     ┌───┐     ┌───┐     ┌───┐                       │     │
│  │    │ A │◄───►│ B │◄───►│ C │◄───►│ D │                       │     │
│  │    └─┬─┘     └─┬─┘     └─┬─┘     └─┬─┘                       │     │
│  │      │    ╲    │    ╱    │    ╲    │                         │     │
│  │      │     ╲   │   ╱     │     ╲   │                         │     │
│  │      ▼      ╲  ▼  ╱      ▼      ╲  ▼                         │     │
│  │    ┌───┐     ┌───┐     ┌───┐     ┌───┐                       │     │
│  │    │ E │◄───►│ F │◄───►│ G │◄───►│ H │                       │     │
│  │    └───┘     └───┘     └───┘     └───┘                       │     │
│  │                                                               │     │
│  │    → Module boundaries disappear and become entangled        │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  2. Deployment Fear                                                     │
│  - Full redeploy even for small changes                                 │
│  - "No Friday deployments" culture                                      │
│  - Release cycles lengthen (monthly → quarterly)                        │
│                                                                         │
│  3. Scaling Inefficiency                                                │
│  - Order service traffic increases → Must scale entire system           │
│  - Resource waste                                                       │
│                                                                         │
│  4. Technical Debt Accumulation                                         │
│  - Difficulty upgrading frameworks/libraries                            │
│  - Cannot adopt new technologies                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Characteristics of Microservices

### Core Characteristics

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Microservices Core Characteristics                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Single Responsibility                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Each service focuses on one business function                  │   │
│  │  - User Service: User management only                           │   │
│  │  - Order Service: Order processing only                         │   │
│  │  - Payment Service: Payment processing only                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. Independent Deployment                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  v1.0    v1.1    v1.2    ← User Service updates                 │   │
│  │  ────────────────────────────────────────────────                │   │
│  │  v2.3                     ← Order Service (no impact)           │   │
│  │  ──────────────────                                              │   │
│  │  v1.5                     ← Product Service (no impact)         │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  3. Technology Diversity (Polyglot)                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │User Service │  │Order Service│  │ ML Service  │              │   │
│  │  │   (Java)    │  │  (Node.js)  │  │  (Python)   │              │   │
│  │  │  PostgreSQL │  │    MySQL    │  │   MongoDB   │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  │                                                                  │   │
│  │  Can choose optimal technology for each service                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  4. Fault Isolation                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Payment Service ✗ (failure)                                    │   │
│  │                                                                  │   │
│  │  User Service ✓   Order Service ✓   Product Service ✓          │   │
│  │  (continues)      (limited func)    (continues)                 │   │
│  │                                                                  │   │
│  │  → Graceful Degradation possible                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Size

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     "How Small is Micro?"                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Qualitative criteria rather than quantitative:                         │
│                                                                         │
│  ✓ Ownable and manageable by one team                                   │
│    - Amazon's "Two-Pizza Team" (6-8 people)                             │
│                                                                         │
│  ✓ Can be rewritten in a few weeks                                      │
│    - Too big makes refactoring/replacement difficult                    │
│                                                                         │
│  ✓ Focus on a single business function                                  │
│    - "Order Service" shouldn't handle user management                   │
│                                                                         │
│  ✓ Independently deployable                                             │
│    - Deploy without changing other services                             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────      │
│                                                                         │
│  Too small: (Nano-service)                                              │
│  - Excessive network calls                                              │
│  - Exploding operational complexity                                     │
│  - Distributed system overhead                                          │
│                                                                         │
│  Too large: (Distributed Monolith)                                      │
│  - Loses microservice benefits                                          │
│  - High coupling                                                        │
│  - Cannot deploy independently                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Defining Service Boundaries

### Domain-Driven Design (DDD)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DDD and Microservices                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Core Concept: Bounded Context                                          │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      E-Commerce Domain                            │ │
│  │                                                                   │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │ │
│  │  │  Sales Context  │  │Inventory Context│  │Shipping Context │   │ │
│  │  │                 │  │                 │  │                 │   │ │
│  │  │  - Order        │  │  - Product      │  │  - Shipment     │   │ │
│  │  │  - Customer     │  │  - Stock        │  │  - Delivery     │   │ │
│  │  │  - Shopping Cart│  │  - Warehouse    │  │  - Carrier      │   │ │
│  │  │                 │  │                 │  │                 │   │ │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │ │
│  │           │                    │                    │            │ │
│  │           ▼                    ▼                    ▼            │ │
│  │     Order Service      Inventory Service     Shipping Service    │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Bounded Context = Good candidate for microservice boundary             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Ubiquitous Language

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Language Differences by Context                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  "Customer" means different things in each Context:                     │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  Sales Context  │  │Support Context  │  │ Billing Context │        │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤        │
│  │                 │  │                 │  │                 │        │
│  │  Customer:      │  │  Customer:      │  │  Customer:      │        │
│  │  - Purchase     │  │  - Ticket       │  │  - Payment      │        │
│  │    history      │  │    history      │  │    methods      │        │
│  │  - Preferences  │  │  - Satisfaction │  │  - Billing      │        │
│  │  - Cart         │  │  - Inquiries    │  │    address      │        │
│  │                 │  │                 │  │  - Payment      │        │
│  │                 │  │                 │  │    history      │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
│                                                                         │
│  Same term, different meanings → Each Context maintains its own model   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────      │
│                                                                         │
│  Anti-pattern: Shared "God Customer" model across all Contexts          │
│                                                                         │
│  class Customer {                                                      │
│    // Sales attributes                                                 │
│    purchaseHistory, preferences, cart...                               │
│    // Support attributes                                               │
│    tickets, satisfaction, inquiries...                                 │
│    // Billing attributes                                               │
│    paymentMethods, billingAddress, invoices...                        │
│    // ... endlessly growing model                                       │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Context Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Relationships Between Contexts                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │   ┌──────────────┐    Upstream    ┌──────────────┐              │   │
│  │   │   Order      │ ─────────────► │  Inventory   │              │   │
│  │   │   Context    │   (depends on) │   Context    │              │   │
│  │   └──────────────┘                └──────────────┘              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Relationship Patterns:                                                 │
│                                                                         │
│  1. Customer-Supplier                                                   │
│     - Upstream reflects Downstream's requirements                       │
│     - Collaborative relationship                                        │
│                                                                         │
│  2. Conformist                                                          │
│     - Downstream uses Upstream's model as-is                            │
│     - Upstream refuses change requests                                  │
│                                                                         │
│  3. Anti-Corruption Layer (ACL)                                         │
│  ┌─────────────┐    ┌─────┐    ┌─────────────┐                        │
│  │   Legacy    │───►│ ACL │───►│  New        │                        │
│  │   System    │    └─────┘    │  Service    │                        │
│  └─────────────┘  (translation └─────────────┘                        │
│                    layer)                                               │
│                                                                         │
│  4. Open Host Service                                                   │
│     - Provides well-documented public API                               │
│     - Supports multiple consumers                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Decomposition Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Service Decomposition Strategies                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. By Business Capability                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Reflect organizational structure (Leverage Conway's Law)       │   │
│  │                                                                  │   │
│  │  Marketing Team ──► Marketing Service                           │   │
│  │  Order Team     ──► Order Service                               │   │
│  │  Shipping Team  ──► Shipping Service                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. By Subdomain                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Core Domain: Core business competitiveness                     │   │
│  │    → Develop in-house, highest quality                          │   │
│  │                                                                  │   │
│  │  Supporting Domain: Supports the core                           │   │
│  │    → Internal development or customization                      │   │
│  │                                                                  │   │
│  │  Generic Domain: General functionality                          │   │
│  │    → External solutions or standard implementation              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  3. By Change Frequency                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Frequent change ◄──────────────────────────► Rarely changes    │   │
│  │                                                                  │   │
│  │  Promotion     Order     Inventory    User Auth                 │   │
│  │  Service       Service   Service      Service                   │   │
│  │                                                                  │   │
│  │  → Separate things with different change frequencies            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Database per Service

### Principle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Database per Service Principle                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Rule: Each service has its own database                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │User Service │  │Order Service│  │Product Svc  │              │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │   │
│  │         │                │                │                      │   │
│  │         ▼                ▼                ▼                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │  User DB    │  │  Order DB   │  │ Product DB  │              │   │
│  │  │ (PostgreSQL)│  │  (MySQL)    │  │ (MongoDB)   │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  │                                                                  │   │
│  │  ✗ Direct DB access forbidden!                                  │   │
│  │                                                                  │   │
│  │  Order Service ──✗──► User DB (forbidden)                       │   │
│  │  Order Service ──✓──► User Service API (allowed)                │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Reasons:                                                               │
│  - Maintain loose coupling                                              │
│  - Independent schema changes possible                                  │
│  - Optimal DB selection per service                                     │
│  - Independent scaling possible                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Sharing Patterns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Inter-Service Data Sharing Patterns                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. API Call (Synchronous)                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Order Service                      User Service                 │   │
│  │       │                                  │                       │   │
│  │       │──GET /users/123 ────────────────►│                       │   │
│  │       │◄──{ name: "Kim", ... } ──────────│                       │   │
│  │       │                                  │                       │   │
│  │  - When real-time data is needed                                 │   │
│  │  - Creates dependency                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. Event-Based (Asynchronous)                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  User Service        Event Bus          Order Service            │   │
│  │       │                  │                   │                   │   │
│  │  Create user             │                   │                   │   │
│  │       │──UserCreated────►│                   │                   │   │
│  │       │                  │──UserCreated─────►│                   │   │
│  │       │                  │                   │ Store local copy  │   │
│  │       │                  │                   │                   │   │
│  │  - Data replication                                              │   │
│  │  - Eventual consistency                                          │   │
│  │  - Loose coupling                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  3. CQRS (Command Query Responsibility Segregation)                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ┌─────────────┐    Events      ┌────────────────────────┐      │   │
│  │  │Command Side │──────────────►│      Read Model        │      │   │
│  │  │(Each service)│              │(Joined views, search   │      │   │
│  │  └─────────────┘              │ optimized)              │      │   │
│  │                               └────────────────────────┘      │   │
│  │                                                                  │   │
│  │  - Read/write separation                                        │   │
│  │  - Complex query optimization                                   │   │
│  │  - Combined with event sourcing                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Distributed Transaction Handling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Distributed Data Consistency                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem: Need to deduct inventory when creating order                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Order Service          Inventory Service                        │   │
│  │  ┌──────────────┐      ┌──────────────┐                         │   │
│  │  │ Create order │      │ Deduct stock │                         │   │
│  │  │ (success)    │      │ (failure!)   │                         │   │
│  │  └──────────────┘      └──────────────┘                         │   │
│  │                                                                  │   │
│  │  → Data inconsistency! (order exists but stock not deducted)    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Solutions:                                                             │
│                                                                         │
│  1. Saga Pattern (Compensating Transactions)                            │
│     - Choreography: Event chain                                         │
│     - Orchestration: Central control                                    │
│     (Details in 10_Data_Consistency_Patterns.md)                        │
│                                                                         │
│  2. Accept Eventual Consistency                                         │
│     - Allow temporary inconsistency based on business requirements      │
│     - Background reconciliation process                                 │
│                                                                         │
│  3. Outbox Pattern                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Order Service                                                   │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  BEGIN TRANSACTION                                        │   │   │
│  │  │    INSERT INTO orders (...)                              │   │   │
│  │  │    INSERT INTO outbox (event: OrderCreated)              │   │   │
│  │  │  COMMIT                                                   │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                          │                                       │   │
│  │                          ▼                                       │   │
│  │  Background Worker: outbox → Message Queue                      │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Inter-Service Communication

### Synchronous Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Synchronous Communication: REST vs gRPC              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  REST (HTTP/JSON)                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Order Service                      Product Service              │   │
│  │       │                                  │                       │   │
│  │       │──GET /products/123 ─────────────►│                       │   │
│  │       │  Content-Type: application/json  │                       │   │
│  │       │                                  │                       │   │
│  │       │◄─{ "id": 123, "name": "...",    │                       │   │
│  │       │    "price": 1000 } ──────────────│                       │   │
│  │       │                                  │                       │   │
│  │  Pros: Simple, easy debugging, universal                         │   │
│  │  Cons: Text-based (overhead), lacks type safety                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  gRPC (HTTP/2 + Protocol Buffers)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Order Service                      Product Service              │   │
│  │       │                                  │                       │   │
│  │       │──GetProduct(id: 123) ───────────►│                       │   │
│  │       │  (Binary, HTTP/2 multiplexing)   │                       │   │
│  │       │                                  │                       │   │
│  │       │◄─Product { id, name, price } ────│                       │   │
│  │       │  (Strongly typed)                │                       │   │
│  │       │                                  │                       │   │
│  │  Pros: Fast, type-safe, bidirectional streaming                  │   │
│  │  Cons: Binary (hard to debug), limited direct browser calls      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────┬─────────────────────┬─────────────────────┐         │
│  │   Feature    │        REST         │        gRPC         │         │
│  ├──────────────┼─────────────────────┼─────────────────────┤         │
│  │ Protocol     │ HTTP/1.1 or 2       │ HTTP/2              │         │
│  │ Format       │ JSON (text)         │ Protobuf (binary)   │         │
│  │ Performance  │ Moderate            │ Fast (~7x)          │         │
│  │ Type Safety  │ Runtime             │ Compile time        │         │
│  │ Streaming    │ Limited             │ Bidirectional       │         │
│  │ Browser      │ Direct possible     │ Needs gRPC-Web      │         │
│  │ Use Case     │ External APIs       │ Internal services   │         │
│  └──────────────┴─────────────────────┴─────────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Asynchronous Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Asynchronous Event-Based Communication               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│  │   Order     │     │   Message   │     │  Inventory  │              │
│  │   Service   │     │    Broker   │     │   Service   │              │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘              │
│         │                   │                   │                      │
│  Create order               │                   │                      │
│         │                   │                   │                      │
│         │──OrderCreated────►│                   │                      │
│         │                   │──OrderCreated────►│                      │
│  (Return immediately)       │                   │ Deduct stock         │
│         │                   │                   │                      │
│         │                   │◄─StockReserved────│                      │
│         │◄─StockReserved────│                   │                      │
│         │                   │                   │                      │
│                                                                         │
│  Advantages:                                                            │
│  - Loose coupling between services                                      │
│  - Messages preserved during service failure                            │
│  - Easy load distribution and scaling                                   │
│                                                                         │
│  Disadvantages:                                                         │
│  - Increased complexity                                                 │
│  - Difficult debugging                                                  │
│  - Eventual consistency                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Communication Pattern Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Communication Pattern Selection                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  When to use Synchronous Communication:                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - When immediate response is needed (queries, validation)      │   │
│  │  - When request-response pattern is natural                     │   │
│  │  - Simple inter-service calls                                   │   │
│  │                                                                  │   │
│  │  Examples: Login auth, product info lookup, balance check       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  When to use Asynchronous Communication:                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - When task completion takes time                               │   │
│  │  - When fault isolation is important                             │   │
│  │  - When event broadcast is needed                                │   │
│  │  - When load leveling is needed                                  │   │
│  │                                                                  │   │
│  │  Examples: Email sending, notifications, log collection,        │   │
│  │  analytics                                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Hybrid:                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Client ──REST──► API Gateway ──Event──► Internal Services      │   │
│  │                                                                  │   │
│  │  - External: REST (simple, universal)                           │   │
│  │  - Internal: Event/gRPC (performance, loose coupling)           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Microservices Adoption Strategy

### Strangler Fig Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Strangler Fig Pattern                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: Proxy in front of monolith                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              ┌─────────┐                                        │   │
│  │  Client ────►│  Proxy  │────► Monolith (100%)                  │   │
│  │              └─────────┘                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Stage 2: Extract some features to microservices                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              ┌─────────┐    ┌─────────────────┐                 │   │
│  │  Client ────►│  Proxy  │───►│ User Service    │ (10%)          │   │
│  │              │         │    └─────────────────┘                 │   │
│  │              │         │    ┌─────────────────┐                 │   │
│  │              │         │───►│ Monolith (90%)  │                 │   │
│  │              └─────────┘    └─────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Stage 3: Gradual migration                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              ┌─────────┐    ┌─────────────────┐                 │   │
│  │  Client ────►│  Proxy  │───►│ User Service    │                 │   │
│  │              │         │───►│ Order Service   │                 │   │
│  │              │         │───►│ Product Service │ (70%)          │   │
│  │              │         │    └─────────────────┘                 │   │
│  │              │         │───►│ Monolith (30%)  │                 │   │
│  │              └─────────┘    └─────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Stage 4: Monolith removal complete                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              ┌─────────┐    ┌─────────────────┐                 │   │
│  │  Client ────►│  Proxy  │───►│ User Service    │                 │   │
│  │              │         │───►│ Order Service   │                 │   │
│  │              │         │───►│ Product Service │ (100%)         │   │
│  │              │         │───►│ Payment Service │                 │   │
│  │              └─────────┘    └─────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Microservices Readiness Checklist

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Microservices Readiness Check                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Organization/Culture:                                                  │
│  ☐ Is DevOps culture established?                                       │
│  ☐ Is the team ready to own and operate services?                       │
│  ☐ Do you have automated CI/CD pipelines?                               │
│  ☐ Is there an on-call/incident response process?                       │
│                                                                         │
│  Infrastructure:                                                        │
│  ☐ Do you have container/orchestration experience? (Docker, K8s)        │
│  ☐ Is monitoring/logging infrastructure ready?                          │
│  ☐ Can you operate a service mesh or API Gateway?                       │
│  ☐ Do you have distributed tracing tools?                               │
│                                                                         │
│  Development:                                                           │
│  ☐ Are domains clearly defined?                                         │
│  ☐ Do you have API design/versioning standards?                         │
│  ☐ Is test automation level sufficient?                                 │
│  ☐ Do you understand distributed system patterns?                       │
│                                                                         │
│  ⚠️  If the above items are not ready, microservices may                │
│     only increase complexity and cost.                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Practice 1: Service Decomposition

Decompose an online bookstore system into microservices:
- Identify core Bounded Contexts
- Define each service's responsibilities
- Design relationships between services (Context Mapping)
- Database separation strategy

### Practice 2: Communication Method Selection

Choose the appropriate sync/async communication method for the following scenarios:
1. Stock check when loading product detail page
2. Email sending after order completion
3. Inventory deduction after payment processing
4. User profile lookup
5. Login event logging

### Practice 3: Migration Plan

Create a plan to transition an existing monolith e-commerce system to microservices:
- Apply Strangler Fig pattern
- Choose the first service to separate and explain why
- Step-by-step migration roadmap

---

**Previous**: [Message Systems Comparison](./12_Message_System_Comparison.md) | **Next**: [Microservices Patterns](./14_Microservices_Patterns.md)

---

## References

- "Building Microservices" - Sam Newman
- "Domain-Driven Design" - Eric Evans
- "Microservices Patterns" - Chris Richardson
- Martin Fowler's Microservices Guide
- Netflix Tech Blog
- Uber Engineering Blog
