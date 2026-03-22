# API Design Principles

**Previous**: [Concurrency & Parallelism](./12_Concurrency_and_Parallelism.md) | **Next**: [Version Control and Collaboration](./14_Version_Control_Workflows.md)

> **Topic**: Programming
> **Lesson**: 13 of 16
> **Prerequisites**: Object-oriented programming, HTTP basics, JSON/XML formats

## Learning Objectives

After completing this lesson, you will be able to:

1. Design RESTful APIs with proper resource URIs, HTTP methods, and status codes
2. Implement pagination, filtering, and sorting for collection endpoints
3. Compare REST, gRPC, and GraphQL and select the appropriate style for a given use case
4. Apply core API design principles: consistency, least surprise, idempotency, and backward compatibility
5. Implement API versioning strategies (URL, header, query parameter) and manage breaking changes
6. Secure APIs using API keys, OAuth 2.0, and JWT-based authentication
7. Design internal library APIs with minimal surface area, encapsulation, and fluent interfaces
8. Document APIs using OpenAPI/Swagger specifications with comprehensive error code references

---

## Introduction

An **API (Application Programming Interface)** is a contract that defines how software components communicate. Whether you're designing a library function, a REST endpoint, or a system interface, good API design makes your software easier to use, maintain, and evolve.

Poor API design leads to:
- Confused users who misuse your interface
- Brittle code that breaks with minor changes
- Support burden from unclear documentation
- Integration difficulties for consumers

Great API design is invisible—users accomplish their goals without friction.

## What Is an API?

An API is an **abstraction** that hides implementation details while exposing functionality:

```python
# Bad: exposing implementation details
user_data = database.execute_raw_sql("SELECT * FROM users WHERE id = ?", user_id)

# Good: abstraction through API
user = user_service.get_user_by_id(user_id)
```

### Types of APIs

1. **Library/Module APIs**: Functions and classes in code libraries
2. **Web APIs**: HTTP-based services (REST, GraphQL, gRPC)
3. **Operating System APIs**: System calls, file I/O, process management
4. **Hardware APIs**: Device drivers, firmware interfaces

This lesson focuses primarily on web APIs and library APIs, as they represent the most common design challenges.

## REST API Design

**REST (Representational State Transfer)** is an architectural style for web APIs that uses HTTP methods and URIs to represent resources.

### Core Principles

#### 1. Resources and URIs

Resources are **nouns**, not verbs. URIs identify resources; HTTP methods indicate actions.

**Good URI Design**:
```
GET    /users              # List users
GET    /users/123          # Get user 123
POST   /users              # Create user
PUT    /users/123          # Update user 123
DELETE /users/123          # Delete user 123
GET    /users/123/orders   # Get orders for user 123
```

**Poor URI Design**:
```
GET    /getUsers           # Verb in URI
POST   /createUser         # Verb in URI
GET    /users/delete/123   # Action in URI path
GET    /user_orders?id=123 # Inconsistent naming
```

**Naming Conventions**:
- Use **plural nouns** for collections (`/users`, not `/user`)
- Use **lowercase** with hyphens (`/order-items`, not `/OrderItems`)
- Represent **hierarchies** with path segments (`/users/123/addresses/456`)

#### 2. HTTP Methods

Use HTTP methods semantically:

| Method | Purpose | Idempotent? | Safe? |
|--------|---------|-------------|-------|
| GET | Retrieve resource | Yes | Yes |
| POST | Create resource | No | No |
| PUT | Replace resource | Yes | No |
| PATCH | Partial update | No | No |
| DELETE | Remove resource | Yes | No |

**Idempotent**: Multiple identical requests have the same effect as a single request
**Safe**: Request doesn't modify server state

```javascript
// Express.js example
const express = require('express');
const app = express();

app.get('/users/:id', (req, res) => {
  const user = userService.findById(req.params.id);
  res.json(user);
});

app.post('/users', (req, res) => {
  const user = userService.create(req.body);
  res.status(201).json(user);
});

app.put('/users/:id', (req, res) => {
  const user = userService.replace(req.params.id, req.body);
  res.json(user);
});

app.patch('/users/:id', (req, res) => {
  const user = userService.update(req.params.id, req.body);
  res.json(user);
});

app.delete('/users/:id', (req, res) => {
  userService.delete(req.params.id);
  res.status(204).send();
});
```

#### 3. HTTP Status Codes

Use appropriate status codes to communicate outcomes:

**2xx Success**:
- `200 OK`: Standard success response
- `201 Created`: Resource created (POST)
- `204 No Content`: Success with no response body (DELETE)

**4xx Client Errors**:
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Authenticated but not authorized
- `404 Not Found`: Resource doesn't exist
- `409 Conflict`: Resource state conflict
- `422 Unprocessable Entity`: Validation error

**5xx Server Errors**:
- `500 Internal Server Error`: Generic server error
- `503 Service Unavailable`: Temporary outage

```python
# Flask example
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()

    if not data or 'email' not in data:
        return jsonify({'error': 'Email is required'}), 400

    if user_exists(data['email']):
        return jsonify({'error': 'User already exists'}), 409

    user = create_user_in_db(data)
    return jsonify(user), 201

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = find_user(user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify(user), 200
```

#### 4. Pagination

For collections, implement pagination to prevent overwhelming responses:

**Offset-based pagination**:
```
GET /users?offset=20&limit=10

Response:
{
  "data": [...],
  "pagination": {
    "offset": 20,
    "limit": 10,
    "total": 150
  }
}
```

**Cursor-based pagination** (better for real-time data):
```
GET /users?cursor=eyJpZCI6MTIzfQ==&limit=10

Response:
{
  "data": [...],
  "next_cursor": "eyJpZCI6MTMzfQ=="
}
```

#### 5. Filtering, Sorting, Searching

Provide query parameters for flexible data retrieval:

```
GET /users?role=admin&sort=-created_at&search=john
```

```java
// Java Spring Boot example
@RestController
@RequestMapping("/users")
public class UserController {

    @GetMapping
    public ResponseEntity<Page<User>> getUsers(
            @RequestParam(required = false) String role,
            @RequestParam(required = false) String search,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(defaultValue = "id") String sortBy,
            @RequestParam(defaultValue = "ASC") Sort.Direction direction
    ) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortBy));
        Page<User> users = userService.findUsers(role, search, pageable);
        return ResponseEntity.ok(users);
    }
}
```

#### 6. HATEOAS

**HATEOAS (Hypermedia As The Engine Of Application State)** includes links to related resources in responses:

```json
{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com",
  "_links": {
    "self": { "href": "/users/123" },
    "orders": { "href": "/users/123/orders" },
    "addresses": { "href": "/users/123/addresses" }
  }
}
```

This makes APIs self-documenting and discoverable.

## RPC-Style APIs

**RPC (Remote Procedure Call)** APIs model actions as function calls rather than resources.

### gRPC

gRPC uses **Protocol Buffers** for efficient binary serialization:

```protobuf
// user.proto
syntax = "proto3";

service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc CreateUser (CreateUserRequest) returns (User);
  rpc ListUsers (ListUsersRequest) returns (ListUsersResponse);
}

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
}

message GetUserRequest {
  int32 id = 1;
}
```

```python
# Python gRPC server
import grpc
from concurrent import futures
import user_pb2
import user_pb2_grpc

class UserService(user_pb2_grpc.UserServiceServicer):
    def GetUser(self, request, context):
        user = find_user(request.id)
        if not user:
            context.abort(grpc.StatusCode.NOT_FOUND, 'User not found')
        return user_pb2.User(id=user.id, name=user.name, email=user.email)

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
user_pb2_grpc.add_UserServiceServicer_to_server(UserService(), server)
server.add_insecure_port('[::]:50051')
server.start()
```

### When to Use RPC vs REST

**Use RPC (gRPC) when**:
- High performance and low latency are critical
- You need bidirectional streaming
- Internal microservice communication
- Type safety is important

**Use REST when**:
- Public-facing APIs
- Broad client compatibility (browsers, mobile)
- Human-readable debugging
- Caching and standard HTTP tooling

## GraphQL

**GraphQL** is a query language that allows clients to request exactly the data they need.

### Schema Definition

```graphql
type User {
  id: ID!
  name: String!
  email: String!
  orders: [Order!]!
}

type Order {
  id: ID!
  total: Float!
  items: [OrderItem!]!
}

type Query {
  user(id: ID!): User
  users(limit: Int, offset: Int): [User!]!
}

type Mutation {
  createUser(name: String!, email: String!): User!
  updateUser(id: ID!, name: String, email: String): User!
}
```

### Queries and Mutations

```javascript
// Client query - request only needed fields
const query = `
  query GetUser($id: ID!) {
    user(id: $id) {
      name
      email
      orders {
        total
      }
    }
  }
`;

// Server resolver (Node.js)
const resolvers = {
  Query: {
    user: (parent, { id }, context) => {
      return context.db.findUserById(id);
    },
  },
  User: {
    orders: (user, args, context) => {
      return context.db.findOrdersByUserId(user.id);
    },
  },
  Mutation: {
    createUser: (parent, { name, email }, context) => {
      return context.db.createUser({ name, email });
    },
  },
};
```

### The N+1 Problem

GraphQL can cause performance issues when resolving nested fields:

```graphql
query {
  users {
    name
    orders {  # N+1: One query per user
      total
    }
  }
}
```

**Solution: DataLoader** batches and caches requests:

```javascript
const DataLoader = require('dataloader');

const orderLoader = new DataLoader(async (userIds) => {
  const orders = await db.findOrdersByUserIds(userIds);
  return userIds.map(id => orders.filter(o => o.userId === id));
});

const resolvers = {
  User: {
    orders: (user, args, { orderLoader }) => {
      return orderLoader.load(user.id);
    },
  },
};
```

### When GraphQL Shines

**Use GraphQL when**:
- Clients need flexible data fetching (mobile, web dashboards)
- You want to avoid over-fetching or under-fetching
- Rapid frontend iteration without backend changes

**Avoid GraphQL when**:
- Simple CRUD operations (REST is simpler)
- File uploads (GraphQL has poor file handling)
- Caching is critical (HTTP caching works better with REST)

## API Design Principles

### 1. Consistency

Consistency reduces cognitive load. Establish conventions and follow them:

**Naming conventions**:
```
# Consistent
GET /users
GET /orders
GET /products

# Inconsistent
GET /users
GET /getOrders
GET /product_list
```

**Response formats**:
```json
// Consistent success response
{
  "data": { ... },
  "meta": { ... }
}

// Consistent error response
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Email is required",
    "fields": { "email": "This field is required" }
  }
}
```

### 2. Principle of Least Surprise

APIs should behave as users expect. Avoid surprising behavior:

```python
# Surprising: method name suggests read-only, but mutates state
def get_user_or_create(email):
    user = find_user(email)
    if not user:
        user = create_user(email)  # Surprise!
    return user

# Clear: name indicates possible mutation
def find_or_create_user(email):
    user = find_user(email)
    if not user:
        user = create_user(email)
    return user
```

### 3. Idempotency

Idempotent operations can be safely retried without side effects:

```python
# Idempotent: PUT replaces entire resource
PUT /users/123
{ "name": "John", "email": "john@example.com" }

# Not idempotent: PATCH might increment
PATCH /users/123
{ "login_count": 5 }  # What if retried?

# Better: server-controlled increments
POST /users/123/login
```

### 4. Backward Compatibility

Never break existing clients. Evolve APIs carefully:

**Safe changes**:
- Adding optional fields
- Adding new endpoints
- Deprecating fields (with grace period)

**Breaking changes**:
- Removing fields
- Changing field types
- Renaming fields
- Changing validation rules

```javascript
// Backward compatible: new optional field
{
  "name": "John",
  "email": "john@example.com",
  "phone": "555-1234"  // New, optional
}

// Breaking: removed field
{
  "name": "John"
  // "email" removed - breaks clients!
}
```

## Versioning Strategies

When breaking changes are necessary, version your API:

### 1. URL Versioning
```
GET /v1/users
GET /v2/users
```
**Pros**: Clear, easy to route
**Cons**: URL changes, caching issues

### 2. Header Versioning
```
GET /users
Accept: application/vnd.myapi.v2+json
```
**Pros**: Clean URLs
**Cons**: Less visible, harder to test in browser

### 3. Query Parameter
```
GET /users?version=2
```
**Pros**: Easy to test
**Cons**: Mixes versioning with filtering

**Recommendation**: URL versioning for major versions, feature flags for minor changes.

## Authentication and Authorization

### API Keys
```
GET /users
X-API-Key: abc123xyz
```
Simple but less secure. Use for low-stakes internal APIs.

### OAuth 2.0
```
GET /users
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```
Industry standard for delegated authorization. Use for third-party integrations.

### JWT (JSON Web Tokens)
```javascript
// Node.js JWT example
const jwt = require('jsonwebtoken');

// Create token
const token = jwt.sign(
  { userId: 123, role: 'admin' },
  process.env.SECRET_KEY,
  { expiresIn: '1h' }
);

// Verify token
function authenticate(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).send('Unauthorized');

  try {
    const payload = jwt.verify(token, process.env.SECRET_KEY);
    req.user = payload;
    next();
  } catch (err) {
    res.status(403).send('Invalid token');
  }
}
```

## Rate Limiting and Throttling

Protect your API from abuse:

```python
# Python Flask rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/search')
@limiter.limit("10 per minute")
def search():
    return perform_expensive_search()
```

**Response headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1614556800
```

Return `429 Too Many Requests` when limit exceeded.

## Error Responses

Provide consistent, helpful error messages:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "email",
        "message": "Email is required"
      },
      {
        "field": "age",
        "message": "Age must be at least 18"
      }
    ],
    "request_id": "req_abc123"
  }
}
```

```cpp
// C++ library API error handling
class ApiException : public std::exception {
private:
    std::string message_;
    int code_;
public:
    ApiException(int code, const std::string& msg)
        : code_(code), message_(msg) {}

    int code() const { return code_; }
    const char* what() const noexcept override {
        return message_.c_str();
    }
};

class UserService {
public:
    User getUser(int id) {
        if (id < 0) {
            throw ApiException(400, "Invalid user ID");
        }
        auto user = db.findById(id);
        if (!user) {
            throw ApiException(404, "User not found");
        }
        return user;
    }
};
```

## Documentation

Great APIs have great documentation.

### OpenAPI (Swagger)

```yaml
# openapi.yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
paths:
  /users/{userId}:
    get:
      summary: Get user by ID
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        email:
          type: string
          format: email
```

**Benefits**:
- Auto-generated interactive documentation
- Client SDK generation
- Validation

### Essential Documentation Elements

1. **Getting Started**: Quick example to make first API call
2. **Authentication**: How to obtain and use credentials
3. **Rate Limits**: Throttling rules and headers
4. **Error Codes**: Comprehensive list with examples
5. **SDKs**: Client libraries for popular languages
6. **Changelog**: Version history and migration guides

## Internal API Design (Libraries and Modules)

### Minimal Surface Area

Expose only what's necessary:

```java
// Bad: exposing internals
public class UserService {
    public Database db;  // Exposed!
    public void validateEmail(String email) { ... }  // Public but internal
}

// Good: minimal interface
public class UserService {
    private Database db;

    public User getUser(int id) { ... }
    public User createUser(UserData data) { ... }

    private void validateEmail(String email) { ... }  // Private
}
```

### Encapsulation

Hide implementation details:

```python
# Bad: implementation exposed
class ShoppingCart:
    def __init__(self):
        self.items = []  # Direct access to internal list

    def add_item(self, item):
        self.items.append(item)

cart = ShoppingCart()
cart.items.append(invalid_item)  # Can bypass validation!

# Good: encapsulated
class ShoppingCart:
    def __init__(self):
        self._items = []

    def add_item(self, item):
        self._validate_item(item)
        self._items.append(item)

    def get_items(self):
        return self._items.copy()  # Return copy, not reference
```

### Fluent Interfaces and Builder Pattern

Make APIs pleasant to use:

```javascript
// Fluent interface
const query = new QueryBuilder()
  .select('name', 'email')
  .from('users')
  .where('age', '>', 18)
  .orderBy('name')
  .limit(10)
  .build();

// Builder pattern
const user = new UserBuilder()
  .withName('John Doe')
  .withEmail('john@example.com')
  .withRole('admin')
  .build();
```

### Consistent Naming and Parameter Ordering

```python
# Consistent: resource first, then options
user_service.get_user(user_id, include_deleted=False)
order_service.get_order(order_id, include_items=True)

# Inconsistent
user_service.get_user(include_deleted=False, user_id=123)
order_service.find_order(123, True)  # What does True mean?
```

## Exercises

### Exercise 1: Design a REST API for a Bookstore

Design a REST API for an online bookstore with the following requirements:
- Browse books (with filtering by genre, author)
- Search books
- User accounts and authentication
- Shopping cart
- Order placement
- Order history

Provide:
1. List of endpoints with HTTP methods
2. Sample request/response bodies
3. Status codes for each endpoint
4. Pagination strategy

### Exercise 2: API Review

Review this API design and identify violations of good API design principles:

```
GET /getBook?id=123
POST /createBook?title=NewBook&author=JohnDoe
GET /updateBook?id=123&title=UpdatedTitle
DELETE /books/remove/123
GET /books?page=1&limit=1000
POST /users/login  (returns 200 even if credentials are wrong)
```

List what's wrong and propose improvements.

### Exercise 3: Versioning Strategy

Your API currently has this endpoint:
```
GET /users/123
Response: { "name": "John", "email": "john@example.com" }
```

You need to add a breaking change: split `name` into `first_name` and `last_name`. Design a versioning strategy that:
1. Doesn't break existing clients
2. Allows new clients to use the improved structure
3. Provides a migration path

### Exercise 4: Error Handling Design

Design a comprehensive error response format for a banking API that needs to handle:
- Validation errors (multiple fields)
- Insufficient funds
- Account locked
- Rate limiting
- Server errors

Provide JSON examples for each error type.

### Exercise 5: Library API Design

Design a fluent API for a SQL query builder in your preferred language that supports:
- SELECT with multiple columns
- FROM with table name
- WHERE with multiple conditions (AND/OR)
- JOIN operations
- ORDER BY
- LIMIT and OFFSET

Example usage:
```
query.select('id', 'name')
     .from('users')
     .where('age > ?', 18)
     .orderBy('name')
     .limit(10)
```

## Summary

Good API design is about empathy for your users:

- **REST**: Resource-oriented, uses HTTP semantics, best for public APIs
- **RPC/gRPC**: Action-oriented, efficient, best for internal microservices
- **GraphQL**: Flexible querying, best for data-heavy client applications
- **Consistency**: Predictable naming, structure, and behavior
- **Idempotency**: Safe retries for PUT and DELETE
- **Versioning**: Manage changes without breaking clients
- **Documentation**: Make APIs self-explanatory with examples
- **Error Handling**: Clear, actionable error messages

Remember: APIs are products. Treat your API consumers as customers—invest in their experience.

---

**Previous**: [Concurrency & Parallelism](./12_Concurrency_and_Parallelism.md) | **Next**: [Version Control and Collaboration](./14_Version_Control_Workflows.md)
