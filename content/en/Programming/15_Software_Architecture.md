# Software Architecture Basics

**Previous**: [Version Control and Collaboration](./14_Version_Control_Workflows.md) | **Next**: [Developer Practices and Ethics](./16_Developer_Practices.md)

> **Topic**: Programming
> **Lesson**: 15 of 16
> **Prerequisites**: Object-oriented programming, design patterns, system design fundamentals

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between architecture (system-level, strategic) and design (component-level, tactical) decisions
2. Evaluate when to use monolithic architecture versus microservices based on team size, complexity, and scaling needs
3. Organize code using layered (N-tier) architecture with clear separation between presentation, business logic, and data access
4. Apply hexagonal architecture (ports and adapters) to decouple domain logic from infrastructure concerns
5. Implement event-driven communication patterns including event sourcing and CQRS
6. Compare MVC, MVP, and MVVM patterns for UI organization and select the appropriate one for a given framework
7. Write Architectural Decision Records (ADRs) to document and communicate design rationale
8. Analyze trade-offs using the CAP theorem and quality attributes (performance, scalability, availability, maintainability)

---

## Introduction

**Software architecture** is the set of high-level structural decisions that shape how a system is built. While design focuses on individual components (classes, functions), architecture focuses on the **big picture**:

- How components are organized
- How they communicate
- What boundaries exist
- How the system scales, evolves, and handles failures

Good architecture:
- Makes the system **easy to understand** (cognitive load)
- Makes the system **easy to change** (adaptability)
- Makes the system **resilient** to failures
- Aligns with **business goals** (cost, time-to-market, scalability)

Poor architecture creates **technical debt** that slows development over time.

## Architecture vs Design

| Aspect | Architecture | Design |
|--------|--------------|--------|
| **Scope** | System-level | Component-level |
| **Decisions** | Strategic, hard to change | Tactical, easier to change |
| **Examples** | Monolith vs microservices, database choice | Class structure, algorithm choice |
| **Impact** | Long-term, affects entire team | Short-term, affects module |
| **When** | Early in project | Throughout project |

**Example**:
- **Architecture decision**: "We'll use microservices with event-driven communication"
- **Design decision**: "The user service will use the repository pattern"

## Monolithic Architecture

A **monolith** is a single deployable unit containing all application logic.

### Structure

```
monolith-app/
├── controllers/       # HTTP handlers
├── services/          # Business logic
├── repositories/      # Data access
├── models/            # Domain entities
└── main.py            # Entry point
```

**Deployment**: One artifact (JAR, executable, Docker image)

### Example (Python Flask Monolith)

```python
# app.py
from flask import Flask, jsonify, request
from database import db
from models import User, Order

app = Flask(__name__)

# User endpoints
@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    user = User(name=data['name'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201

# Order endpoints
@app.route('/orders', methods=['GET'])
def get_orders():
    orders = Order.query.all()
    return jsonify([o.to_dict() for o in orders])

@app.route('/orders', methods=['POST'])
def create_order():
    data = request.get_json()
    order = Order(user_id=data['user_id'], total=data['total'])
    db.session.add(order)
    db.session.commit()
    return jsonify(order.to_dict()), 201

if __name__ == '__main__':
    app.run()
```

All features live in one codebase and share one database.

### Advantages

**1. Simplicity**: One codebase, one deployment, one server
```bash
# Deploy entire app
docker build -t myapp .
docker run -p 8000:8000 myapp
```

**2. Easy debugging**: All code in one process, use single debugger
**3. Consistent data**: Transactions across features (ACID guarantees)
**4. Performance**: No network calls between modules (in-memory function calls)
**5. Development speed**: For small teams, monoliths are fastest to build

### Disadvantages

**1. Scaling limitations**: Must scale entire app, even if only one feature needs resources
```
# If search is CPU-heavy but checkout is not,
# you still scale both together
docker run --replicas=10 myapp
```

**2. Technology lock-in**: Entire app uses same language, framework, database
```python
# Hard to rewrite one feature in Go while rest is Python
```

**3. Deployment coupling**: Deploy entire app for small changes (risky, slow)
```bash
# Fix typo in one endpoint → redeploy entire app
```

**4. Code coupling**: Features can become tightly coupled over time
```python
# User service directly calls Order service
def create_user(data):
    user = User(**data)
    db.session.add(user)
    order_service.create_welcome_order(user.id)  # Tight coupling
    db.session.commit()
```

**5. Team coordination**: Large teams step on each other's toes (merge conflicts, deploy conflicts)

### When to Use Monoliths

✅ **Good fit**:
- Small to medium teams (< 20 developers)
- Early-stage products (MVP, validation phase)
- Simple domains with low complexity
- CRUD applications with limited traffic

❌ **Poor fit**:
- Large teams (> 50 developers)
- High-scale systems (millions of requests/day)
- Diverse technology needs (ML + web + real-time)

**Remember**: Most successful companies started with monoliths. Don't prematurely optimize for scale.

## Microservices Architecture

**Microservices** split the application into independent, deployable services organized around business capabilities.

### Structure

```
company-system/
├── user-service/          # Manages users
│   ├── src/
│   ├── database/
│   └── Dockerfile
├── order-service/         # Manages orders
│   ├── src/
│   ├── database/
│   └── Dockerfile
├── payment-service/       # Processes payments
│   ├── src/
│   ├── database/
│   └── Dockerfile
└── api-gateway/           # Routes requests
    ├── src/
    └── Dockerfile
```

Each service:
- Has its **own database**
- Is **independently deployable**
- Communicates via **network** (HTTP, gRPC, message queue)
- Is owned by a **single team**

### Example (Node.js Microservices)

**User Service**:
```javascript
// user-service/server.js
const express = require('express');
const app = express();

app.get('/users/:id', (req, res) => {
  const user = db.findUser(req.params.id);
  res.json(user);
});

app.post('/users', (req, res) => {
  const user = db.createUser(req.body);
  res.status(201).json(user);
});

app.listen(3001);
```

**Order Service**:
```javascript
// order-service/server.js
const express = require('express');
const axios = require('axios');
const app = express();

app.post('/orders', async (req, res) => {
  // Call user service to validate user
  const user = await axios.get(`http://user-service:3001/users/${req.body.userId}`);

  if (!user.data) {
    return res.status(404).send('User not found');
  }

  const order = db.createOrder(req.body);
  res.status(201).json(order);
});

app.listen(3002);
```

**API Gateway**:
```javascript
// api-gateway/server.js
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();

app.use('/users', createProxyMiddleware({ target: 'http://user-service:3001' }));
app.use('/orders', createProxyMiddleware({ target: 'http://order-service:3002' }));
app.use('/payments', createProxyMiddleware({ target: 'http://payment-service:3003' }));

app.listen(8080);
```

### Advantages

**1. Independent scaling**: Scale only what needs scaling
```bash
# Scale order service 10x, user service 2x
kubectl scale deployment order-service --replicas=10
kubectl scale deployment user-service --replicas=2
```

**2. Technology diversity**: Each service can use best-fit technology
```
user-service:   Node.js + MongoDB
order-service:  Python + PostgreSQL
search-service: Go + Elasticsearch
```

**3. Team autonomy**: Teams own services end-to-end (reduce coordination)
**4. Fault isolation**: One service failure doesn't crash entire system
```javascript
// Order service falls back if payment service is down
try {
  await paymentService.charge(order);
} catch (err) {
  await queue.enqueue('retry-payment', order);
  return res.status(202).send('Payment pending');
}
```

**5. Faster deployments**: Deploy one service without touching others

### Disadvantages

**1. Complexity**: Network calls, service discovery, distributed tracing
```python
# Simple in monolith
user = get_user(user_id)

# Complex in microservices
response = requests.get(f'http://user-service/users/{user_id}')
if response.status_code != 200:
    # Handle network error, timeout, service down...
user = response.json()
```

**2. Distributed transactions**: No ACID across services
```javascript
// In monolith: atomic transaction
db.transaction(() => {
  createOrder(order);
  decrementInventory(order.items);
  chargePayment(order.total);
});

// In microservices: eventual consistency
await orderService.createOrder(order);
await inventoryService.decrementStock(order.items);  // Might fail
await paymentService.charge(order.total);  // Might fail
// Need saga pattern or compensation logic
```

**3. Network latency**: Every service call adds 10-100ms
**4. Operational overhead**: More deployments, monitoring, logs to manage
**5. Data duplication**: Services may cache data from other services

### When NOT to Use Microservices

❌ **Avoid microservices if**:
- You're building an MVP (unproven product)
- Team is small (< 10 developers)
- You don't have DevOps expertise (monitoring, orchestration)
- Domain boundaries are unclear (services will change frequently)

**Famous quote**: "You must be this tall to use microservices" — Martin Fowler

**Start with a monolith**. Extract microservices later when pain points emerge (team scaling, performance bottlenecks).

## Layered (N-Tier) Architecture

**Layered architecture** organizes code into horizontal layers with clear responsibilities.

### Classic 3-Tier Architecture

```
┌─────────────────────────┐
│   Presentation Layer    │  (UI, controllers, API endpoints)
├─────────────────────────┤
│   Business Logic Layer  │  (Services, domain logic)
├─────────────────────────┤
│   Data Access Layer     │  (Repositories, ORM)
├─────────────────────────┤
│   Database              │  (PostgreSQL, MongoDB)
└─────────────────────────┘
```

### Example (Java Spring Boot)

```java
// Presentation Layer: Controller
@RestController
@RequestMapping("/users")
public class UserController {
    @Autowired
    private UserService userService;

    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        User user = userService.findById(id);
        return ResponseEntity.ok(user);
    }

    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody UserDTO dto) {
        User user = userService.createUser(dto);
        return ResponseEntity.status(201).body(user);
    }
}

// Business Logic Layer: Service
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private EmailService emailService;

    public User createUser(UserDTO dto) {
        // Business logic
        if (userRepository.existsByEmail(dto.getEmail())) {
            throw new UserAlreadyExistsException();
        }

        User user = new User(dto.getName(), dto.getEmail());
        userRepository.save(user);

        emailService.sendWelcomeEmail(user.getEmail());
        return user;
    }

    public User findById(Long id) {
        return userRepository.findById(id)
            .orElseThrow(() -> new UserNotFoundException(id));
    }
}

// Data Access Layer: Repository
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    boolean existsByEmail(String email);
}
```

### Strict vs Relaxed Layering

**Strict layering**: Each layer only calls the layer directly below
```
Presentation → Business Logic → Data Access
```

**Relaxed layering**: Layers can skip levels
```
Presentation → Data Access (for simple reads, bypassing business logic)
```

**Trade-off**: Strict layering enforces separation but adds boilerplate. Relaxed layering is pragmatic but can lead to logic leaking into presentation.

### Advantages

- **Separation of concerns**: Clear responsibilities
- **Testability**: Mock layers independently
- **Replaceability**: Swap data layer (SQL → NoSQL) without touching business logic

### Disadvantages

- **Performance**: Extra layers add overhead
- **Rigidity**: Forces logic into layers even when unnatural

## Hexagonal Architecture (Ports and Adapters)

**Hexagonal architecture** (Alistair Cockburn, 2005) places the **domain** at the center, surrounded by **ports** (interfaces) and **adapters** (implementations).

### Structure

```
       ┌───────────────────┐
       │    Adapters       │
       │  (HTTP, CLI, DB)  │
       └─────────┬─────────┘
                 │
       ┌─────────▼─────────┐
       │      Ports        │
       │   (Interfaces)    │
       └─────────┬─────────┘
                 │
       ┌─────────▼─────────┐
       │  Domain/Core      │
       │ (Business Logic)  │
       └───────────────────┘
```

**Key idea**: Domain logic doesn't depend on infrastructure (databases, frameworks). Infrastructure depends on domain.

### Example (Python)

**Domain (core business logic)**:
```python
# domain/user.py
class User:
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

    def change_email(self, new_email):
        if '@' not in new_email:
            raise ValueError('Invalid email')
        self.email = new_email
```

**Port (interface)**:
```python
# ports/user_repository.py
from abc import ABC, abstractmethod

class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, user_id):
        pass

    @abstractmethod
    def save(self, user):
        pass
```

**Adapter (implementation)**:
```python
# adapters/postgres_user_repository.py
from ports.user_repository import UserRepository
from domain.user import User

class PostgresUserRepository(UserRepository):
    def __init__(self, db_connection):
        self.db = db_connection

    def find_by_id(self, user_id):
        row = self.db.execute('SELECT * FROM users WHERE id = ?', user_id)
        return User(row['id'], row['name'], row['email'])

    def save(self, user):
        self.db.execute('UPDATE users SET name = ?, email = ? WHERE id = ?',
                        user.name, user.email, user.id)
```

**Use case (application service)**:
```python
# use_cases/change_email.py
class ChangeEmailUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def execute(self, user_id, new_email):
        user = self.user_repository.find_by_id(user_id)
        user.change_email(new_email)
        self.user_repository.save(user)
```

**Adapter (HTTP)**:
```python
# adapters/flask_app.py
from flask import Flask, request, jsonify
from use_cases.change_email import ChangeEmailUseCase
from adapters.postgres_user_repository import PostgresUserRepository

app = Flask(__name__)
user_repo = PostgresUserRepository(db_connection)

@app.route('/users/<int:user_id>/email', methods=['PUT'])
def change_email(user_id):
    use_case = ChangeEmailUseCase(user_repo)
    use_case.execute(user_id, request.json['email'])
    return jsonify({'status': 'ok'})
```

### Dependency Rule

**Dependencies point inward**: Outer layers depend on inner layers, never the reverse.

```
Adapters (HTTP, DB) → Ports → Domain
```

**Benefit**: Domain logic is **testable** without infrastructure:
```python
# Test without database
class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self.users = {}

    def find_by_id(self, user_id):
        return self.users[user_id]

    def save(self, user):
        self.users[user.id] = user

def test_change_email():
    repo = InMemoryUserRepository()
    repo.save(User(1, 'John', 'john@example.com'))

    use_case = ChangeEmailUseCase(repo)
    use_case.execute(1, 'newemail@example.com')

    user = repo.find_by_id(1)
    assert user.email == 'newemail@example.com'
```

## Clean Architecture

**Clean Architecture** (Robert C. Martin, 2012) generalizes hexagonal architecture with explicit layers:

```
┌──────────────────────────────────────┐
│  Frameworks & Drivers (DB, Web, UI)  │  (Outermost)
├──────────────────────────────────────┤
│  Interface Adapters (Controllers)    │
├──────────────────────────────────────┤
│  Use Cases (Application Logic)       │
├──────────────────────────────────────┤
│  Entities (Business Logic)           │  (Innermost)
└──────────────────────────────────────┘
```

### Layers

1. **Entities**: Core business rules (User, Order)
2. **Use Cases**: Application-specific logic (CreateUser, PlaceOrder)
3. **Interface Adapters**: Convert data for use cases (Controllers, Presenters)
4. **Frameworks & Drivers**: External tools (Web framework, database, UI)

### Dependency Rule

**Dependencies only point inward**. Inner layers know nothing about outer layers.

```cpp
// C++ Example: Entities (innermost)
// entities/Order.h
class Order {
private:
    int id_;
    double total_;
    std::vector<Item> items_;
public:
    Order(int id, double total) : id_(id), total_(total) {}

    void addItem(const Item& item) {
        items_.push_back(item);
        total_ += item.price;
    }

    double getTotal() const { return total_; }
};

// Use Cases
// use_cases/PlaceOrder.h
class PlaceOrder {
private:
    OrderRepository* repository_;  // Interface (port)
public:
    PlaceOrder(OrderRepository* repo) : repository_(repo) {}

    void execute(int userId, const std::vector<Item>& items) {
        Order order(generateId(), 0);
        for (const auto& item : items) {
            order.addItem(item);
        }
        repository_->save(order);
    }
};

// Interface Adapter
// adapters/HttpController.cpp
void OrderController::placeOrder(HttpRequest& req, HttpResponse& res) {
    auto items = parseItems(req.body);
    PlaceOrder useCase(orderRepository);
    useCase.execute(req.userId, items);
    res.send(201, "Order placed");
}
```

### Independence

Clean architecture achieves independence from:
- **Frameworks**: No lock-in to Rails, Spring, etc.
- **UI**: Swap web UI for CLI without changing logic
- **Database**: Swap SQL for NoSQL
- **External agencies**: Third-party APIs are behind interfaces

## Event-Driven Architecture

**Event-Driven Architecture (EDA)** uses **events** (state changes) to trigger communication between components.

### Components

- **Event**: Something that happened (`UserRegistered`, `OrderPlaced`)
- **Producer**: Emits events
- **Event Bus**: Routes events to consumers
- **Consumer**: Reacts to events

### Example (Microservices with Events)

```javascript
// Node.js with RabbitMQ
const amqp = require('amqplib');

// Producer: User Service
class UserService {
    async registerUser(data) {
        const user = await db.createUser(data);

        // Emit event
        const connection = await amqp.connect('amqp://localhost');
        const channel = await connection.createChannel();
        channel.publish('events', 'user.registered', Buffer.from(JSON.stringify({
            userId: user.id,
            email: user.email,
        })));

        return user;
    }
}

// Consumer: Email Service
class EmailService {
    async start() {
        const connection = await amqp.connect('amqp://localhost');
        const channel = await connection.createChannel();
        channel.consume('user.registered', (msg) => {
            const event = JSON.parse(msg.content.toString());
            this.sendWelcomeEmail(event.email);
            channel.ack(msg);
        });
    }

    sendWelcomeEmail(email) {
        console.log(`Sending welcome email to ${email}`);
    }
}
```

**Decoupling**: User service doesn't know about email service. They communicate through events.

### Event Sourcing

**Event Sourcing** stores **state as a sequence of events** rather than current state.

```python
# Traditional: Store current state
user = { 'id': 1, 'email': 'new@example.com', 'name': 'John' }
db.save(user)

# Event Sourcing: Store events
events = [
    {'type': 'UserCreated', 'data': {'name': 'John', 'email': 'old@example.com'}},
    {'type': 'EmailChanged', 'data': {'email': 'new@example.com'}},
]
db.save_events(events)

# Rebuild state by replaying events
def get_user_state(user_id):
    events = db.get_events(user_id)
    user = {}
    for event in events:
        if event['type'] == 'UserCreated':
            user = event['data']
        elif event['type'] == 'EmailChanged':
            user['email'] = event['data']['email']
    return user
```

**Benefits**: Full audit trail, time travel, replay for debugging
**Drawbacks**: Complexity, eventual consistency

### CQRS (Command Query Responsibility Segregation)

**CQRS** separates **writes** (commands) from **reads** (queries).

```
Commands (writes) → Write Model → Event Store
                                      ↓
                                  Event Bus
                                      ↓
                                  Read Model (optimized for queries)
```

**Example**:
- **Write Model**: Normalized SQL for consistency
- **Read Model**: Denormalized Elasticsearch for fast searches

## MVC, MVP, MVVM

Patterns for organizing UI code.

### MVC (Model-View-Controller)

```
User Input → Controller → Model
                 ↓          ↓
              View ←────────┘
```

**Model**: Business logic, data
**View**: UI rendering
**Controller**: Handles input, updates model

```python
# Flask MVC
@app.route('/users/<int:user_id>')
def show_user(user_id):  # Controller
    user = User.query.get(user_id)  # Model
    return render_template('user.html', user=user)  # View
```

### MVP (Model-View-Presenter)

```
User Input → Presenter ↔ Model
                 ↕
              View
```

View is passive; Presenter handles all logic.

### MVVM (Model-View-ViewModel)

```
User Input → ViewModel ↔ Model
               ↕ (data binding)
            View
```

**ViewModel** exposes data and commands for View. Popular in frameworks with data binding (React, Vue, Angular).

## Architectural Decision Records (ADR)

**ADRs** document important architectural decisions.

### Template

```markdown
# ADR-001: Use Microservices Architecture

## Status
Accepted

## Context
We need to scale independently as user base grows.
Current monolith deploys entire app for small changes, causing downtime.
Team has grown to 30 developers, causing merge conflicts.

## Decision
We will adopt microservices architecture with these services:
- User Service
- Order Service
- Payment Service
- Notification Service

Each service will have its own database.
Communication via REST APIs and RabbitMQ events.

## Consequences
### Positive
- Independent scaling
- Team autonomy
- Faster deployments

### Negative
- Increased complexity
- Need for service mesh, distributed tracing
- Eventual consistency challenges

## Alternatives Considered
- **Modular monolith**: Keep monolith but enforce module boundaries
  - Rejected: Doesn't solve scaling or deployment issues
```

## Quality Attributes

Architecture optimizes for **quality attributes** (non-functional requirements):

- **Performance**: Response time, throughput
- **Scalability**: Handle growing load
- **Availability**: Uptime percentage (99.9% = 8.76 hours downtime/year)
- **Maintainability**: Ease of changing code
- **Security**: Resistance to attacks
- **Testability**: Ease of testing

**Trade-offs**: You can't optimize all attributes. Prioritize based on business needs.

## CAP Theorem

In distributed systems, you can have **at most two** of:

- **Consistency**: All nodes see the same data at the same time
- **Availability**: Every request gets a response (success or failure)
- **Partition Tolerance**: System continues despite network partitions

```
     Consistency
        /   \
       /     \
   CA /       \ CP
     /         \
    /           \
Availability—Partition Tolerance
         AP
```

**Examples**:
- **CA**: Traditional RDBMS (PostgreSQL) — consistent and available, but not partition-tolerant
- **CP**: MongoDB, HBase — consistent and partition-tolerant, but may be unavailable during partitions
- **AP**: Cassandra, DynamoDB — available and partition-tolerant, but eventually consistent

**In practice**: Network partitions happen, so choose CP or AP.

## Exercises

### Exercise 1: Choose Architecture for Scenarios

For each scenario, recommend an architecture (monolith, microservices, serverless) and justify:

1. Startup building an MVP for a task management app (3 developers, 6-month timeline)
2. E-commerce company with 100 developers, 1M daily users, needs to scale checkout independently
3. Internal tool for data processing (runs monthly batch jobs, 2 developers)

### Exercise 2: Design a Layered Architecture

Design a 3-tier layered architecture for a blogging platform with:
- User authentication
- Post creation, editing, deletion
- Comments
- Search

List the layers and what components belong in each.

### Exercise 3: Refactor to Hexagonal Architecture

Given this tightly coupled code, refactor to hexagonal architecture:

```python
from flask import Flask, request
import psycopg2

app = Flask(__name__)

@app.route('/users', methods=['POST'])
def create_user():
    conn = psycopg2.connect("dbname=mydb user=postgres")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, email) VALUES (%s, %s)",
                   (request.json['name'], request.json['email']))
    conn.commit()
    return {'status': 'ok'}
```

Define:
1. Domain entity
2. Port (interface)
3. Adapter (implementation)
4. Use case

### Exercise 4: Document an Architectural Decision

Write an ADR for one of these decisions:
1. Choosing PostgreSQL over MongoDB for user data
2. Adopting GraphQL instead of REST
3. Using server-side rendering vs client-side rendering

### Exercise 5: Analyze CAP Trade-offs

Your company is building a real-time collaborative document editor (like Google Docs). Users must see each other's edits instantly, and the system must work even if some servers are unreachable.

Which of CAP would you prioritize? Would you choose CP or AP? Explain your reasoning and potential trade-offs.

## Summary

Software architecture is about making **intentional trade-offs** to meet business goals:

- **Monolith**: Simple, fast to build, scales vertically — start here
- **Microservices**: Complex, scales horizontally, team autonomy — evolve to this when needed
- **Layered Architecture**: Separation of concerns, testability
- **Hexagonal/Clean Architecture**: Domain independence, testability, flexibility
- **Event-Driven**: Decoupling, scalability, eventual consistency
- **MVC/MVP/MVVM**: UI organization patterns
- **ADRs**: Document decisions and trade-offs
- **Quality Attributes**: Prioritize what matters (performance, scalability, maintainability)
- **CAP Theorem**: You can't have it all in distributed systems

**Golden rule**: Choose the simplest architecture that meets your needs. You can always evolve later.

---

**Previous**: [Version Control and Collaboration](./14_Version_Control_Workflows.md) | **Next**: [Developer Practices and Ethics](./16_Developer_Practices.md)
