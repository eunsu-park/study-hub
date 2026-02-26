# Load Balancing

**Previous**: [Network Fundamentals Review](./03_Network_Fundamentals_Review.md) | **Next**: [Reverse Proxy & API Gateway](./05_Reverse_Proxy_API_Gateway.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a load balancer does and why it is essential for horizontally scaled systems
2. Distinguish between Layer 4 and Layer 7 load balancers and select the appropriate type for a given scenario
3. Compare distribution algorithms (round robin, weighted round robin, least connections, IP hash) and their trade-offs
4. Describe how sticky sessions work and identify when they are necessary versus harmful
5. Implement health check mechanisms to detect and route around unhealthy backend servers
6. Design a highly available load balancer setup using active-passive or active-active configurations

**Difficulty**: ⭐⭐⭐
**Estimated Study Time**: 2-3 hours
**Prerequisites**: [03_Network_Fundamentals_Review.md](./03_Network_Fundamentals_Review.md)

---

When millions of requests pour into a service every second, no single server can handle the flood alone. A load balancer acts as the traffic cop that decides which backend server should process each incoming request, ensuring no single machine is overwhelmed while others sit idle. Getting this distribution right is the difference between a responsive application and one that collapses under its own success.

## Table of Contents

1. [What is Load Balancing?](#1-what-is-load-balancing)
2. [L4 vs L7 Load Balancer](#2-l4-vs-l7-load-balancer)
3. [Distribution Algorithms](#3-distribution-algorithms)
4. [Sticky Session](#4-sticky-session)
5. [Health Checks](#5-health-checks)
6. [High Availability Configuration](#6-high-availability-configuration)
7. [Practice Problems](#7-practice-problems)
8. [References](#8-references)

---

## 1. What is Load Balancing?

### 1.1 Definition

Load balancing is a technique that distributes incoming network traffic across multiple servers to improve system availability and performance.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Load Balancing Concept                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Without load balancer:                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client 1 ─────┐                                           │ │
│  │  Client 2 ─────┼─────────▶ Server 1 (Overloaded!)          │ │
│  │  Client 3 ─────┤                                           │ │
│  │     ...        │                                           │ │
│  │  Client N ─────┘                                           │ │
│  │                                                            │ │
│  │  Problem: Single server overload, service down on failure  │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  With load balancer:                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client 1 ─────┐         ┌────────▶ Server 1               │ │
│  │  Client 2 ─────┼─▶ [LB] ─┼────────▶ Server 2               │ │
│  │  Client 3 ─────┤         └────────▶ Server 3               │ │
│  │     ...        │                                           │ │
│  │  Client N ─────┘                                           │ │
│  │                                                            │ │
│  │  Solution: Distributed load, high availability, horizontal │ │
│  │           scaling possible                                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Load Balancer Roles

```
┌─────────────────────────────────────────────────────────────────┐
│                 Load Balancer Key Functions                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Traffic Distribution                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Distribute requests evenly across servers                │ │
│  │ • Can apply weights based on server capacity               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. High Availability                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Detect server failures and automatically switch traffic  │ │
│  │ • Users don't notice failures                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Scalability                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Easy to add/remove servers                               │ │
│  │ • Zero-downtime server replacement possible                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. SSL Termination                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Handle HTTPS at LB, backend uses HTTP                    │ │
│  │ • Reduce server load, centralize certificate management    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  5. Session Management                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Sticky Session to keep same server                       │ │
│  │ • Ensure session data consistency                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. L4 vs L7 Load Balancer

### 2.1 L4 Load Balancer (Transport Layer)

```
┌─────────────────────────────────────────────────────────────────┐
│                  L4 Load Balancer                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OSI Layer 4 (Transport) - Operates at TCP/UDP level            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Checks only packet header info:                          │ │
│  │  ┌─────────────────────────────────────────────────┐       │ │
│  │  │ Source IP    │ Dest IP     │ Source   │ Dest   │       │ │
│  │  │ 203.0.113.50 │ 10.0.0.100  │ Port     │ Port   │       │ │
│  │  │              │             │ 54321    │ 80     │       │ │
│  │  └─────────────────────────────────────────────────┘       │ │
│  │                       ↑                                    │ │
│  │                  Routes using this info                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  How it works:                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client ──▶ LB (10.0.0.100:80)                             │ │
│  │                    │                                       │ │
│  │           ┌────────┴────────┐                              │ │
│  │           │ IP/Port based   │                              │ │
│  │           │ routing decision│                              │ │
│  │           └────────┬────────┘                              │ │
│  │                    │                                       │ │
│  │          ┌─────────┼─────────┐                             │ │
│  │          ▼         ▼         ▼                             │ │
│  │     Server 1   Server 2   Server 3                         │ │
│  │     10.0.1.1   10.0.1.2   10.0.1.3                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                          │
│  • Fast processing (no packet content inspection)               │
│  • Low resource usage                                           │
│  • Protocol independent (TCP, UDP both work)                    │
│                                                                  │
│  Cons:                                                          │
│  • No application awareness                                     │
│  • Cannot route based on URL                                    │
│  • Cannot make content-based decisions                          │
│                                                                  │
│  Examples: AWS NLB, HAProxy (TCP mode), LVS                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 L7 Load Balancer (Application Layer)

```
┌─────────────────────────────────────────────────────────────────┐
│                  L7 Load Balancer                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OSI Layer 7 (Application) - Operates at HTTP/HTTPS level       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Analyzes HTTP request content:                            │ │
│  │  ┌─────────────────────────────────────────────────┐       │ │
│  │  │ GET /api/users HTTP/1.1                         │       │ │
│  │  │ Host: api.example.com                           │       │ │
│  │  │ Cookie: session=abc123                          │       │ │
│  │  │ Authorization: Bearer xxx                       │       │ │
│  │  │ Content-Type: application/json                  │       │ │
│  │  └─────────────────────────────────────────────────┘       │ │
│  │           ↑                                                │ │
│  │      Analyzes full content to route                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  URL-based routing:                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  /api/users/*   ──────────▶ User Service                   │ │
│  │  /api/orders/*  ──────────▶ Order Service                  │ │
│  │  /api/products/*──────────▶ Product Service                │ │
│  │  /static/*      ──────────▶ CDN/Static Server              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Host-based routing:                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  api.example.com   ──────▶ API Servers                     │ │
│  │  www.example.com   ──────▶ Web Servers                     │ │
│  │  admin.example.com ──────▶ Admin Servers                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                          │
│  • Smart routing based on content                               │
│  • SSL termination                                              │
│  • Can modify requests/responses                                │
│  • A/B testing, canary deployment                               │
│                                                                  │
│  Cons:                                                          │
│  • Slower than L4 (needs packet analysis)                       │
│  • More resource usage                                          │
│  • Limited to HTTP/HTTPS                                        │
│                                                                  │
│  Examples: AWS ALB, Nginx, HAProxy (HTTP mode), Envoy          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Comparison Summary

| Item | L4 Load Balancer | L7 Load Balancer |
|------|---------------|---------------|
| OSI Layer | 4 (Transport) | 7 (Application) |
| Analysis Target | IP, Port | HTTP headers, URL, cookies, etc. |
| Protocol | TCP, UDP | HTTP, HTTPS, WebSocket |
| Speed | Very Fast | Fast |
| Resources | Low | High |
| Features | Simple distribution | Smart routing, SSL, caching |
| Use Cases | Game servers, DNS, internal comms | Web services, APIs |

---

## 3. Distribution Algorithms

> **Analogy -- Netflix Queue and the Grocery Store**
>
> Imagine a grocery store with five checkout lanes. A **round-robin** load balancer is like a greeter who sends each new customer to the next lane in sequence -- lane 1, lane 2, lane 3, lane 4, lane 5, then back to lane 1. Simple, but it ignores the fact that lane 3 might already have someone with a cart full of items.
>
> A **least-connections** balancer is like a smart greeter who always points you to the lane with the fewest people. And **weighted** distribution is like having some lanes staffed with experienced cashiers who can handle twice the throughput -- so the greeter sends two people there for every one to the slower lanes.
>
> Netflix uses similar logic at massive scale: when you press play, a load balancer decides which of hundreds of streaming servers should send you the video, optimizing for server load, geographic proximity, and available bandwidth -- all in milliseconds.

### 3.1 Round Robin

```
┌─────────────────────────────────────────────────────────────────┐
│                      Round Robin                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Distribute in sequential order"                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Request 1 ──▶ Server A                                    │ │
│  │  Request 2 ──▶ Server B                                    │ │
│  │  Request 3 ──▶ Server C                                    │ │
│  │  Request 4 ──▶ Server A  (cycles back)                     │ │
│  │  Request 5 ──▶ Server B                                    │ │
│  │  Request 6 ──▶ Server C                                    │ │
│  │       ...                                                  │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                          │
│  • Simple implementation                                        │
│  • No state storage needed                                      │
│  • Even distribution (for same-performance servers)             │
│                                                                  │
│  Cons:                                                          │
│  • Doesn't consider server performance differences              │
│  • Doesn't consider connection duration                         │
│                                                                  │
│  Suitable: Servers with same performance, similar request       │
│           processing times                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Weighted Round Robin

```
┌─────────────────────────────────────────────────────────────────┐
│                  Weighted Round Robin                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Assign weights based on server performance"                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Server A: Weight 5  (high performance)                    │ │
│  │  Server B: Weight 3  (medium performance)                  │ │
│  │  Server C: Weight 2  (low performance)                     │ │
│  │                                                            │ │
│  │  10 requests distribution:                                 │ │
│  │  A A A A A B B B C C                                       │ │
│  │  ▲─────────▲─────▲─▲                                       │ │
│  │  5         3     2                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Suitable: Different server performances, gradual new server    │
│           introduction (canary)                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Least Connections

```
┌─────────────────────────────────────────────────────────────────┐
│                    Least Connections                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Distribute to server with fewest connections"                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Current state:                                            │ │
│  │  Server A: 5 connections                                   │ │
│  │  Server B: 3 connections  ◀── New request assigned         │ │
│  │  Server C: 7 connections                                   │ │
│  │                                                            │ │
│  │  After new request:                                        │ │
│  │  Server A: 5 connections                                   │ │
│  │  Server B: 4 connections                                   │ │
│  │  Server C: 7 connections                                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Weighted Least Connections:                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Assign to server with smallest (connections / weight)     │ │
│  │                                                            │ │
│  │  Server A: 5 conn / weight 5 = 1.0                         │ │
│  │  Server B: 3 conn / weight 2 = 1.5                         │ │
│  │  Server C: 4 conn / weight 3 = 1.33                        │ │
│  │                                                            │ │
│  │  → Assign to Server A (1.0 is smallest)                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                          │
│  • Dynamic load distribution                                    │
│  • Effective for requests with varying processing times         │
│                                                                  │
│  Suitable: Varying request processing times, long connections   │
│           (WebSocket)                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Pseudocode:**

```python
def least_connections_route(servers, request):
    # Unlike round-robin (stateless), least-connections must track live
    # connection counts — more accurate but requires atomic updates
    min_server = None
    min_score = float('inf')

    for server in servers:
        if not server.healthy:
            continue
        # Weighted variant: normalize by capacity so a powerful server
        # with 10 connections can still beat a weak one with 3
        score = server.active_connections / server.weight

        if score < min_score:
            min_score = score
            min_server = server

    # Atomically increment BEFORE forwarding — prevents two concurrent
    # requests from both choosing the same "least loaded" server
    min_server.active_connections += 1
    forward(request, min_server)
```

### 3.4 IP Hash

```
┌─────────────────────────────────────────────────────────────────┐
│                       IP Hash                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Hash client IP to always route to same server"                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  hash(Client IP) % server count = server index             │ │
│  │                                                            │ │
│  │  Client 203.0.113.50:                                      │ │
│  │    hash(203.0.113.50) = 12345                              │ │
│  │    12345 % 3 = 0 ──▶ Server A                              │ │
│  │                                                            │ │
│  │  Client 198.51.100.25:                                     │ │
│  │    hash(198.51.100.25) = 67890                             │ │
│  │    67890 % 3 = 1 ──▶ Server B                              │ │
│  │                                                            │ │
│  │  ※ Same IP always routes to same server!                  │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                          │
│  • Guarantees same server without Sticky Session                │
│  • Cache efficiency (same user = same server cache)             │
│                                                                  │
│  Cons:                                                          │
│  • Many users redistributed when adding/removing servers        │
│  • Imbalance in NAT environments (same public IP)               │
│                                                                  │
│  Alternative: Consistent Hashing (minimizes impact of server    │
│              changes)                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Algorithm Comparison Summary

| Algorithm | Operation | Pros | Cons | Use Cases |
|----------|------|------|------|----------|
| Round Robin | Sequential | Simple, even | Ignores performance | Same servers |
| Weighted RR | Weight-based | Reflects performance | Static weights | Various servers |
| Least Conn | Minimum connections | Dynamic distribution | Needs state tracking | Long connections |
| IP Hash | IP hash | Consistency | Possible imbalance | Session persistence |

---

## 4. Sticky Session

### 4.1 What is Sticky Session?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sticky Session                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Always route same user's requests to same server"             │
│                                                                  │
│  Problem (without Sticky Session):                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  User ──Request 1──▶ Server A (login, create session)      │ │
│  │  User ──Request 2──▶ Server B (no session? logged out!)    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution (Sticky Session):                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  User ──Request 1──▶ Server A (create session)             │ │
│  │  User ──Request 2──▶ Server A (same server!)               │ │
│  │  User ──Request 3──▶ Server A (same server!)               │ │
│  │                                                            │ │
│  │           ┌─────────────────────────────────────────┐      │ │
│  │           │ Load Balancer                           │      │ │
│  │           │ User A → Server A (Cookie/IP based)     │      │ │
│  │           │ User B → Server B                       │      │ │
│  │           │ User C → Server A                       │      │ │
│  │           └─────────────────────────────────────────┘      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Sticky Session Implementation Methods

```
┌─────────────────────────────────────────────────────────────────┐
│              Sticky Session Implementation Methods               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Cookie-based                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  First request:                                            │ │
│  │  Client ──────────▶ LB ──────────▶ Server A                │ │
│  │         ◀──────────    ◀──────────                         │ │
│  │    Set-Cookie: SERVERID=server-a                           │ │
│  │                                                            │ │
│  │  Subsequent requests:                                      │ │
│  │  Client ──────────▶ LB (check cookie) ──▶ Server A         │ │
│  │    Cookie: SERVERID=server-a                               │ │
│  │                                                            │ │
│  │  Pros: Most common, reliable                               │ │
│  │  Cons: Doesn't work if cookies disabled                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Source IP-based                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client IP 203.0.113.50 → always Server A                  │ │
│  │  Client IP 198.51.100.25 → always Server B                 │ │
│  │                                                            │ │
│  │  Pros: No cookies needed                                   │ │
│  │  Cons: Issues in NAT, proxy environments                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Application Cookie (Session ID)                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Use app-generated session ID (JSESSIONID, etc.)           │ │
│  │  Cookie: JSESSIONID=abc123 → hash(abc123) → Server A       │ │
│  │                                                            │ │
│  │  Pros: Matches app session                                 │ │
│  │  Cons: No cookie on first request                          │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Sticky Session Problems

```
┌─────────────────────────────────────────────────────────────────┐
│             Sticky Session Problems and Alternatives             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Problems:                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. Load imbalance                                         │ │
│  │     "Heavy" users may cluster on specific servers          │ │
│  │                                                            │ │
│  │  2. Session loss on failure                                │ │
│  │     Server down → all users on that server lose sessions   │ │
│  │                                                            │ │
│  │  3. Difficult Auto Scaling                                 │ │
│  │     Need user redistribution when adding/removing servers  │ │
│  │                                                            │ │
│  │  4. Horizontal scaling limitations                         │ │
│  │     Stateful architecture limitations                      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Alternatives (Stateless Architecture):                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. External session storage (Redis, Memcached)            │ │
│  │     All servers access same session data                   │ │
│  │                                                            │ │
│  │  2. JWT (JSON Web Token)                                   │ │
│  │     Include session info in token, no server storage       │ │
│  │                                                            │ │
│  │  3. Database sessions                                      │ │
│  │     Store sessions in DB (slower but reliable)             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Health Checks

### 5.1 What are Health Checks?

```
┌─────────────────────────────────────────────────────────────────┐
│                      Health Checks                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Periodically check server status to detect failures"          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │           Load Balancer                                    │ │
│  │               │                                            │ │
│  │    ┌──────────┼──────────┐                                 │ │
│  │    │          │          │                                 │ │
│  │    ▼          ▼          ▼                                 │ │
│  │  Server A   Server B   Server C                            │ │
│  │    ✓          ✓          ✗ (failed)                        │ │
│  │                                                            │ │
│  │  Traffic distribution: A, B only (exclude C)               │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Periodically:                                                  │
│  1. Send health check request                                   │
│  2. Check response                                              │
│  3. Accumulate failure count                                    │
│  4. Exclude server when threshold exceeded                      │
│  5. Include again when recovered                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Active vs Passive Health Checks

```
┌─────────────────────────────────────────────────────────────────┐
│              Active vs Passive Health Checks                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Active Health Check (Proactive)                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  LB periodically sends requests to server                  │ │
│  │                                                            │ │
│  │  ┌──────┐   GET /health   ┌──────┐                         │ │
│  │  │  LB  │ ───────────────▶│Server│                         │ │
│  │  │      │ ◀───────────────│      │                         │ │
│  │  └──────┘   200 OK        └──────┘                         │ │
│  │                                                            │ │
│  │  Configuration example:                                    │ │
│  │  • Interval: 10 seconds                                    │ │
│  │  • Timeout: 5 seconds                                      │ │
│  │  • Failure threshold: 3 times                              │ │
│  │  • Recovery threshold: 2 times                             │ │
│  │                                                            │ │
│  │  Pros: Detect failures without traffic                     │ │
│  │  Cons: Additional load, network traffic                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Passive Health Check (Reactive)                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Monitor actual request responses                          │ │
│  │                                                            │ │
│  │  Client ──▶ LB ──▶ Server                                  │ │
│  │                      │                                     │ │
│  │              ┌───────┴───────┐                             │ │
│  │              │ 5xx error?    │                             │ │
│  │              │ Timeout?      │                             │ │
│  │              │ Connect fail? │                             │ │
│  │              └───────────────┘                             │ │
│  │                      │                                     │ │
│  │              Increment failure count                       │ │
│  │                                                            │ │
│  │  Pros: No additional traffic, reflects real conditions     │ │
│  │  Cons: Real users experience errors                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Recommendation: Active + Passive combination                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Health Check Endpoint Design

```
┌─────────────────────────────────────────────────────────────────┐
│              Health Check Endpoint Design                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Simple Health Check                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  GET /health                                               │ │
│  │                                                            │ │
│  │  Response: 200 OK                                          │ │
│  │  { "status": "healthy" }                                   │ │
│  │                                                            │ │
│  │  Purpose: Check process alive                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Detailed Health Check                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  GET /health/details                                       │ │
│  │                                                            │ │
│  │  Response:                                                 │ │
│  │  {                                                         │ │
│  │    "status": "healthy",                                    │ │
│  │    "version": "1.2.3",                                     │ │
│  │    "uptime": "3d 5h 20m",                                  │ │
│  │    "dependencies": {                                       │ │
│  │      "database": "healthy",                                │ │
│  │      "redis": "healthy",                                   │ │
│  │      "external_api": "degraded"                            │ │
│  │    }                                                       │ │
│  │  }                                                         │ │
│  │                                                            │ │
│  │  Purpose: Debugging, monitoring dashboard                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Readiness vs Liveness (Kubernetes)                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Liveness: Is process alive?                               │ │
│  │    GET /health/live                                        │ │
│  │    On failure: Restart container                           │ │
│  │                                                            │ │
│  │  Readiness: Ready to receive traffic?                      │ │
│  │    GET /health/ready                                       │ │
│  │    On failure: Exclude from traffic (don't restart)        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. High Availability Configuration

### 6.1 Active-Passive (Hot Standby)

```
┌─────────────────────────────────────────────────────────────────┐
│                  Active-Passive Configuration                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Standby takes over when one dies"                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Normal state:                                             │ │
│  │                                                            │ │
│  │  Clients ──▶ VIP (10.0.0.100) ──▶ LB Primary (Active)      │ │
│  │                                     │                      │ │
│  │                                     │ Heartbeat            │ │
│  │                                     ▼                      │ │
│  │                               LB Secondary (Standby)       │ │
│  │                                                            │ │
│  │  On failure:                                               │ │
│  │                                                            │ │
│  │  Clients ──▶ VIP (10.0.0.100) ──▶ LB Primary (Down!)       │ │
│  │                    │                                       │ │
│  │                    │ VIP takeover (Failover)                │ │
│  │                    ▼                                       │ │
│  │               LB Secondary (Active)                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Implementation: VRRP, Keepalived, Pacemaker                    │
│  Pros: Simple, resource efficient (Standby just waits)          │
│  Cons: Standby resource waste, brief downtime on switch         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Active-Active

```
┌─────────────────────────────────────────────────────────────────┐
│                   Active-Active Configuration                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Both load balancers handle traffic simultaneously"            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │                      DNS                                   │ │
│  │                       │                                    │ │
│  │            ┌──────────┴──────────┐                         │ │
│  │            │                     │                         │ │
│  │            ▼                     ▼                         │ │
│  │     LB 1 (Active)          LB 2 (Active)                   │ │
│  │     VIP: 10.0.0.100        VIP: 10.0.0.101                 │ │
│  │            │                     │                         │ │
│  │            └──────────┬──────────┘                         │ │
│  │                       │                                    │ │
│  │            ┌──────────┼──────────┐                         │ │
│  │            ▼          ▼          ▼                         │ │
│  │        Server 1   Server 2   Server 3                      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  On failure:                                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  LB 1 (Down!) ──▶ Remove from DNS or                       │ │
│  │                  LB 2 takes over VIP                       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros: 100% resource utilization, higher throughput             │
│  Cons: Complex configuration, state sync needed                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Cloud Load Balancers

```
┌─────────────────────────────────────────────────────────────────┐
│                  Cloud Load Balancers                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AWS:                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • ALB (Application Load Balancer) - L7                     │ │
│  │   URL/host-based routing, WebSocket, HTTP/2                │ │
│  │                                                            │ │
│  │ • NLB (Network Load Balancer) - L4                         │ │
│  │   Ultra-low latency, fixed IP, TCP/UDP                     │ │
│  │                                                            │ │
│  │ • CLB (Classic Load Balancer) - L4/L7 (legacy)             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  GCP:                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • HTTP(S) Load Balancer - L7, global                       │ │
│  │ • TCP/UDP Load Balancer - L4, regional                     │ │
│  │ • Internal Load Balancer - internal traffic                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Advantages:                                                    │
│  • Managed service (reduced operations burden)                  │
│  • Auto scaling, built-in high availability                     │
│  • Integrated with Auto Scaling                                 │
│  • Security features (WAF, DDoS protection)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Problem 1: Choose Load Balancer

Choose appropriate load balancer type (L4/L7) for these scenarios and explain why.

a) gRPC communication between microservices
b) Multi-language website (URL: /ko/, /en/, /jp/)
c) Real-time game server
d) API gateway

### Problem 2: Choose Distribution Algorithm

Choose most suitable distribution algorithm for these situations.

a) All servers same specs, similar request processing times
b) WebSocket-based chat service
c) Gradually introducing new server (canary)
d) Service utilizing server-side caching

### Problem 3: Health Check Design

Design health check endpoint for e-commerce service.

Conditions:
- Depends on database, Redis, payment API
- Kubernetes environment
- Need fast failure detection

### Problem 4: Architecture Design

Design load balancer architecture for service handling 100M daily requests.

Conditions:
- Global users (Asia, North America, Europe)
- 99.99% availability requirement
- Cost optimization

---

## Answers

### Problem 1 Answer

```
a) gRPC comms: L4 (NLB)
   - gRPC is HTTP/2-based but L4 sufficient
   - Low latency, high throughput

b) Multi-language website: L7 (ALB)
   - URL-based routing needed (/ko → Korean server)
   - Host header analysis

c) Game server: L4 (NLB)
   - UDP support needed
   - Minimal latency critical
   - TCP/UDP game protocols

d) API gateway: L7 (ALB)
   - URL/header-based routing
   - Authentication, Rate Limiting
   - SSL termination
```

### Problem 2 Answer

```
a) Round Robin
   - Same servers, similar requests → simple rotation efficient

b) Least Connections
   - WebSocket are long connections
   - Connection count-based distribution effective for load balance

c) Weighted Round Robin
   - New server with low weight (e.g., 10%)
   - Gradually increase weight

d) IP Hash or Consistent Hashing
   - Same user → same server → increased cache hit rate
```

### Problem 3 Answer

```json
// GET /health/ready (Readiness)
{
  "status": "healthy",
  "checks": {
    "database": {
      "status": "healthy",
      "latency_ms": 5
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 1
    },
    "payment_api": {
      "status": "healthy",
      "latency_ms": 50
    }
  }
}

// GET /health/live (Liveness)
{ "status": "healthy" }

Configuration:
- Readiness: interval 5s, failure threshold 2, recovery threshold 1
- Liveness: interval 10s, failure threshold 3

Readiness failure: Exclude from traffic (dependency issues like payment)
Liveness failure: Restart container
```

### Problem 4 Answer

```
Architecture:

1. Global Load Balancing (DNS)
   - AWS Route 53 / Cloudflare (Latency-based)
   - Asia → Seoul region
   - North America → Virginia region
   - Europe → Frankfurt region

2. Per-region configuration:
   ┌─────────────────────────────────┐
   │ CDN (CloudFront/Cloudflare)    │ ← Static content
   └─────────────────────────────────┘
                    │
   ┌─────────────────────────────────┐
   │ L7 Load Balancer (ALB)         │ ← URL routing
   │ - Auto Scaling                 │
   │ - WAF integration              │
   └─────────────────────────────────┘
                    │
   ┌─────────────────────────────────┐
   │ Application Servers            │
   │ (Auto Scaling Group)           │
   └─────────────────────────────────┘

3. Ensure availability:
   - Multi-AZ deployment
   - Health checks: Active + Passive
   - Enable Cross-Zone Load Balancing

4. Cost optimization:
   - Reserved Instances
   - CDN reduces Origin load
   - Auto Scaling per region traffic patterns
```

---

## Hands-On Exercises

### Exercise 1: Load Balancer Algorithm Comparison

Use `examples/System_Design/04_load_balancer.py` to explore load balancer behavior.

**Tasks:**
1. Run all demos and observe the distribution for each algorithm
2. Add a new scenario: 5 servers where one has 3× the weight of others. Compare weighted round-robin vs. least-connections for 1000 requests with random connection durations (1-20s)
3. Implement a new algorithm: **Weighted Least Connections** — combine server weights with active connection counts
4. Test with a mix of long-running (10s) and short (1s) requests. Which algorithm achieves the best balance?

### Exercise 2: Health Check Simulator

Extend the load balancer with health check functionality.

**Tasks:**
1. Add a `health_check()` method that marks servers as healthy/unhealthy based on success/failure counts
2. Implement a configurable threshold: mark unhealthy after N consecutive failures, healthy after M consecutive successes
3. Simulate a scenario where one server starts failing at t=10, recovers at t=30
4. Track metrics: requests served by each server, requests dropped during transitions, total downtime detected

### Exercise 3: Session Affinity with Failover

Implement IP-hash load balancing with graceful failover.

**Tasks:**
1. Start with `IPHashLB` from the example code
2. When a server goes down, redistribute only that server's clients to remaining servers using consistent hashing (reference `07_consistent_hashing.py`)
3. When the server recovers, gradually migrate clients back (not all at once)
4. Measure the percentage of clients that change servers during failover vs. modular hash approach

---

## 8. References

### Tools
- [Nginx](https://nginx.org/) - Web server & load balancer
- [HAProxy](https://www.haproxy.org/) - High-performance load balancer
- [Envoy](https://www.envoyproxy.io/) - Cloud-native proxy

### Documentation
- [AWS Elastic Load Balancing](https://aws.amazon.com/elasticloadbalancing/)
- [GCP Cloud Load Balancing](https://cloud.google.com/load-balancing)
- [Nginx Load Balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/)

### Online Resources
- [High Availability Load Balancers](https://www.nginx.com/blog/nginx-high-availability-with-haproxy/)

---

**Previous**: [Network Fundamentals Review](./03_Network_Fundamentals_Review.md) | **Next**: [Reverse Proxy & API Gateway](./05_Reverse_Proxy_API_Gateway.md)

---

**Document Information**
- Last Updated: 2024
- Difficulty: ⭐⭐⭐
- Estimated Study Time: 2-3 hours
