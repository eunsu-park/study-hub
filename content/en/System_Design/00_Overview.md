# System Design Learning Guide

## Introduction

This folder contains materials for systematically learning System Design. You can learn step-by-step from core concepts to practical patterns needed to design large-scale systems. It supports both technical interview preparation and practical architecture design.

**Target Audience**: Backend developers, system architects, technical interview candidates

---

## Learning Roadmap

```
[Basics]                  [Intermediate]            [Advanced]
  │                         │                         │
  ▼                         ▼                         ▼
System Design Overview ──▶ Load Balancing ─────▶ Distributed Cache
  │                         │                         │
  ▼                         ▼                         ▼
Scalability Basics ─────▶ Reverse Proxy ─────▶ Database Scaling
  │                         │                         │
  ▼                         ▼                         ▼
Network Review ──────────▶ Caching Strategy ───▶ Database Replication
  │                         │                         │
  ▼                         ▼                         ▼
                      API Gateway ─────────▶ Message Queue
                                                 │
                                                 ▼
                                            Microservices
                                                 │
                                                 ▼
                                            Distributed Systems
                                                 │
                                                 ▼
                                            Practical Design
```

---

## Prerequisites

- **Required**
  - Networking basics (HTTP, DNS, TCP/IP) → [Networking/](../Networking/00_Overview.md)
  - Database basics (SQL, transactions) → [PostgreSQL/](../PostgreSQL/00_Overview.md)
  - At least 1 programming language

- **Recommended**
  - Linux basic commands → [Linux/](../Linux/00_Overview.md)
  - Docker basics → [Docker/](../Docker/00_Overview.md)
  - REST API concepts → [Web_Development/](../Web_Development/00_Overview.md)

---

## File List

### Basics (01-03)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [01_System_Design_Overview.md](./01_System_Design_Overview.md) | ⭐ | What is system design, interview criteria, problem approach framework |
| [02_Scalability_Basics.md](./02_Scalability_Basics.md) | ⭐⭐ | Vertical/horizontal scaling, CAP theorem, PACELC |
| [03_Network_Fundamentals_Review.md](./03_Network_Fundamentals_Review.md) | ⭐⭐ | DNS, CDN, HTTP/2/3, REST vs gRPC |

### Load Balancing and Proxy (04-05)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [04_Load_Balancing.md](./04_Load_Balancing.md) | ⭐⭐⭐ | L4/L7 load balancers, distribution algorithms, health checks |
| [05_Reverse_Proxy_API_Gateway.md](./05_Reverse_Proxy_API_Gateway.md) | ⭐⭐⭐ | Reverse proxy, API Gateway, Rate Limiting |

### Caching (06-07)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [06_Caching_Strategies.md](./06_Caching_Strategies.md) | ⭐⭐⭐ | Cache-Aside, Write-Through, cache invalidation |
| [07_Distributed_Cache_Systems.md](./07_Distributed_Cache_Systems.md) | ⭐⭐⭐ | Redis, Memcached, consistent hashing |

### Database Scaling (08-10)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [08_Database_Scaling.md](./08_Database_Scaling.md) | ⭐⭐⭐ | Partitioning, sharding strategies, rebalancing |
| [09_Database_Replication.md](./09_Database_Replication.md) | ⭐⭐⭐ | Leader replication, Quorum, failure recovery |
| [10_Data_Consistency_Patterns.md](./10_Data_Consistency_Patterns.md) | ⭐⭐⭐ | Consistency models, eventual consistency, strong consistency |

### Message Queue (11-12)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [11_Message_Queue_Basics.md](./11_Message_Queue_Basics.md) | ⭐⭐⭐ | Async processing, Kafka, RabbitMQ |
| [12_Message_System_Comparison.md](./12_Message_System_Comparison.md) | ⭐⭐⭐⭐ | Kafka vs RabbitMQ, use cases |

### Microservices (13-14)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [13_Microservices_Basics.md](./13_Microservices_Basics.md) | ⭐⭐⭐⭐ | Monolith vs MSA, service decomposition |
| [14_Microservices_Patterns.md](./14_Microservices_Patterns.md) | ⭐⭐⭐⭐ | Service mesh, Circuit Breaker, Saga |

### Distributed Systems (15-16)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [15_Distributed_Systems_Concepts.md](./15_Distributed_Systems_Concepts.md) | ⭐⭐⭐⭐ | Distributed system properties, failure models |
| [16_Consensus_Algorithms.md](./16_Consensus_Algorithms.md) | ⭐⭐⭐⭐⭐ | Raft, Paxos, leader election |

### Practical Design (17-18)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [17_Design_Example_1.md](./17_Design_Example_1.md) | ⭐⭐⭐ | URL shortener, Pastebin design |
| [18_Design_Example_2.md](./18_Design_Example_2.md) | ⭐⭐⭐⭐ | Chat system, notification system design |

### Observability and Search (19-20)

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [19_Observability_Monitoring.md](./19_Observability_Monitoring.md) | ⭐⭐⭐⭐ | Three pillars, Prometheus, ELK, OpenTelemetry, SLO/SLI |
| [20_Search_Systems.md](./20_Search_Systems.md) | ⭐⭐⭐⭐ | Elasticsearch, inverted index, BM25, scaling search |

---

## Recommended Learning Path

### Phase 1: Build Foundations (1 week)
```
01_System_Design_Overview → 02_Scalability_Basics → 03_Network_Fundamentals_Review
```
Learn core concepts and interview approach to system design.

### Phase 2: Traffic Handling (1 week)
```
04_Load_Balancing → 05_Reverse_Proxy_API_Gateway
```
Learn traffic distribution and API management.

### Phase 3: Master Caching (1 week)
```
06_Caching_Strategies → 07_Distributed_Cache_Systems
```
Take a deep dive into caching, which is core to performance optimization.

### Phase 4: Database Scaling (1-2 weeks)
```
08_Database_Scaling → 09_Database_Replication → 10_Data_Consistency_Patterns
```
Learn DB scaling strategies for large-scale data.

### Phase 5: Message Queues and Microservices (2 weeks)
```
11_Message_Queue_Basics → 12_Message_System_Comparison → 13_Microservices_Basics → 14_Microservices_Patterns
```
Learn async processing and distributed architecture patterns.

### Phase 6: Distributed Systems and Practical Design (2-3 weeks)
```
15_Distributed_Systems_Concepts → 16_Consensus_Algorithms → 17_Design_Example_1 → 18_Design_Example_2
```
Cover distributed systems theory and practical design problems.

### Phase 7: Observability and Search (1-2 weeks)
```
19_Observability_Monitoring → 20_Search_Systems
```
Learn system monitoring, alerting, and search system design.

---

## Interview Preparation Tips

### System Design Interview Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                System Design Interview 4 Steps                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Clarify Requirements (5 min)                                │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Confirm functional requirements                      │  │
│     │ • Non-functional requirements (performance, etc.)      │  │
│     │ • Scale estimation (users, traffic)                    │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  2. Back-of-the-envelope Calculation (5 min)                    │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • QPS (Queries Per Second)                             │  │
│     │ • Storage capacity                                     │  │
│     │ • Bandwidth                                            │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  3. High-Level Design (15-20 min)                               │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Main component diagram                               │  │
│     │ • Data flow                                            │  │
│     │ • API design                                           │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  4. Detailed Design (15-20 min)                                 │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Database schema                                      │  │
│     │ • Scaling strategy                                     │  │
│     │ • Trade-offs discussion                                │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Frequently Asked Topics

| Topic | Key Concepts | Related Files |
|------|----------|----------|
| URL Shortener | Hash function, Base62 | 08, 07 |
| Chat System | WebSocket, message queue | 03, 11 |
| News Feed | Fan-out, caching | 06, 07 |
| Search Engine | Inverted index, sharding | 08 |
| Notification System | Message queue, priority | 11, 12 |
| File Storage | Distributed storage, chunks | 08, 09 |

### Interview Checklist

- [ ] Did you clarify requirements?
- [ ] Did you estimate scale?
- [ ] Did you draw high-level architecture?
- [ ] Did you design data model?
- [ ] Did you identify bottlenecks?
- [ ] Did you propose scaling strategy?
- [ ] Did you discuss trade-offs?

---

## Related Resources

### Links to Other Folders

| Folder | Related Content |
|------|----------|
| [Networking/](../Networking/00_Overview.md) | DNS, HTTP, TCP/IP, network security |
| [PostgreSQL/](../PostgreSQL/00_Overview.md) | Database, transactions, replication |
| [Docker/](../Docker/00_Overview.md) | Containerization, microservices deployment |
| [Linux/](../Linux/00_Overview.md) | Server management, performance monitoring |

### Recommended Books

- **Designing Data-Intensive Applications** - Martin Kleppmann
- **System Design Interview** - Alex Xu
- **Building Microservices** - Sam Newman
- **Web Scalability for Startup Engineers** - Artur Ejsmont

### Online Resources

- [System Design Primer (GitHub)](https://github.com/donnemartin/system-design-primer)
- [Grokking System Design (Educative)](https://www.educative.io/courses/grokking-the-system-design-interview)
- [High Scalability Blog](http://highscalability.com/)
- [ByteByteGo Blog](https://bytebytego.com/)

---

## Learning Tips

1. **Draw Diagrams**: Draw system architectures yourself
2. **Estimation Skills**: Get familiar with QPS and storage capacity calculations
3. **Trade-offs**: Every decision has pros and cons
4. **Real Examples**: Analyze architectures of large services
5. **Interview Practice**: Practice explaining out loud

---

**Document Information**
- Last Updated: January 2026
- Total Study Time: Approximately 6-8 weeks
