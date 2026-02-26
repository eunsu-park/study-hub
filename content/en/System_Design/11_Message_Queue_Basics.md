# Message Queue Fundamentals

**Previous**: [Data Consistency Patterns](./10_Data_Consistency_Patterns.md) | **Next**: [Message System Comparison](./12_Message_System_Comparison.md)

Difficulty: ⭐⭐⭐

## Learning Objectives

1. Distinguish between synchronous and asynchronous communication and identify when each is appropriate
2. Explain the core benefits of message queues in distributed architectures (decoupling, buffering, resilience)
3. Compare queue-based (point-to-point) and topic-based (publish-subscribe) messaging models
4. Describe the three message delivery guarantees (at-most-once, at-least-once, exactly-once) and their trade-offs
5. Implement idempotent message consumers using deduplication keys and idempotency tokens
6. Apply common message queue patterns (work queue, fan-out, request-reply, dead letter queue) to real-world scenarios

---

Every modern web application you use -- from ride-sharing to e-commerce checkout -- relies on message queues behind the scenes to handle work that does not need an immediate response. When you place an order online, the confirmation page appears instantly while payment processing, inventory updates, and email notifications happen asynchronously through message queues. Understanding message queues is the first step toward building systems that remain responsive under heavy load and gracefully survive partial failures.

## Table of Contents

1. [Synchronous vs Asynchronous Communication](#1-synchronous-vs-asynchronous-communication)
2. [Benefits of Message Queues](#2-benefits-of-message-queues)
3. [Queue vs Topic](#3-queue-vs-topic)
4. [Message Delivery Guarantees](#4-message-delivery-guarantees)
5. [Idempotency](#5-idempotency)
6. [Message Queue Patterns](#6-message-queue-patterns)
7. [Practice Problems](#7-practice-problems)

---

## 1. Synchronous vs Asynchronous Communication

### Synchronous Communication

Waiting for a response after making a request.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Synchronous (Request-Response)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐                           ┌──────────┐                   │
│  │ Service A│                           │ Service B│                   │
│  └────┬─────┘                           └────┬─────┘                   │
│       │                                      │                          │
│       │─────── HTTP Request ────────────────►│                          │
│       │                                      │                          │
│       │        (Service A blocked)           │ Processing...            │
│       │        Request thread waiting        │                          │
│       │                                      │                          │
│       │◄────── HTTP Response ───────────────│                          │
│       │                                      │                          │
│       ▼                                      ▼                          │
│  Continue next task                    Processing complete              │
│                                                                         │
│  Characteristics:                                                       │
│  - Immediate response required                                          │
│  - Caller waits for response                                            │
│  - Service A affected when Service B fails                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Asynchronous Communication

Send a message and immediately continue with other tasks.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Asynchronous (Message Queue)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌──────────────┐     ┌──────────┐                   │
│  │ Service A│     │ Message Queue│     │ Service B│                   │
│  │(Producer)│     │              │     │(Consumer)│                   │
│  └────┬─────┘     └──────┬───────┘     └────┬─────┘                   │
│       │                  │                  │                          │
│       │──── Send Msg ───►│                  │                          │
│       │◄─── ACK ─────────│                  │                          │
│       │                  │                  │                          │
│       ▼                  │                  │                          │
│  Continue next task      │◄─── Poll ────────│                          │
│  (no blocking)           │──── Deliver ────►│                          │
│                          │                  │ Processing...            │
│                          │◄─── ACK ─────────│                          │
│                          │                  ▼                          │
│                          │             Processing complete             │
│                                                                         │
│  Characteristics:                                                       │
│  - Sender/receiver independent                                          │
│  - Loose coupling                                                       │
│  - Service A operates normally even when Service B fails                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Communication Methods Comparison

| Characteristic | Synchronous | Asynchronous |
|------|------|--------|
| Wait for response | Required | Not required |
| Coupling | Strong | Loose |
| Failure propagation | Immediate | Isolated |
| Real-time nature | Immediate | Can be delayed |
| Complexity | Low | High |
| Throughput | Limited | High |

---

## 2. Benefits of Message Queues

### Decoupling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Tight Coupling vs Loose Coupling                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Tight Coupling (synchronous):                                         │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐                          │
│  │  A  │────►│  B  │────►│  C  │────►│  D  │                          │
│  └─────┘     └─────┘     └─────┘     └─────┘                          │
│     │                       X Failure!                                  │
│     ▼                                                                   │
│  A also fails! (Wait for B → C fails → entire failure)                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Loose Coupling (asynchronous):                                        │
│                      ┌─────────────┐                                   │
│                      │   Message   │                                   │
│  ┌─────┐            │    Queue    │            ┌─────┐                │
│  │  A  │───publish──►├─────────────┤◄──consume──│  B  │                │
│  └─────┘            │  [msg][msg] │            └─────┘                │
│     │               │  [msg][msg] │               X Failure!           │
│     ▼               └─────────────┘                                    │
│  A continues!                                                          │
│  (Messages stored in queue, processed after B recovers)                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Load Leveling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Load Leveling                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Without Queue (direct call):                                          │
│                                                                         │
│  Request    │    ████                                                  │
│  volume     │    ████ ← Server overload at peak!                       │
│  Processing │ ───████─────────────                                     │
│  capacity   │    ████                                                  │
│             └────────────────────────► Time                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  With Queue (buffering):                                               │
│                                                                         │
│  Request    │    ████                                                  │
│  volume     │    ████                                                  │
│  Queue      │ ───████████████──                                        │
│  size       │    ████████████                                          │
│  Processing │ ═══════════════════════ ← Steady processing              │
│             └────────────────────────► Time                            │
│                                                                         │
│  Queue acts as buffer:                                                 │
│  - Absorb peak traffic                                                 │
│  - Consumer processes at steady rate                                   │
│  - Improved system stability                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Scalability

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Horizontal Scaling                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│              ┌────────────┐                                            │
│              │  Message   │                                            │
│              │   Queue    │                                            │
│              ├────────────┤                                            │
│  Producer───►│ [m1][m2]   │                                            │
│              │ [m3][m4]   │                                            │
│              │ [m5][m6]   │                                            │
│              └─────┬──────┘                                            │
│                    │                                                    │
│         ┌──────────┼──────────┐                                        │
│         ▼          ▼          ▼                                        │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐                                │
│    │Consumer1│ │Consumer2│ │Consumer3│                                │
│    │  (m1)   │ │  (m2)   │ │  (m3)   │                                │
│    └─────────┘ └─────────┘ └─────────┘                                │
│                                                                         │
│  To increase throughput:                                                │
│  - Just add more consumers                                              │
│  - No producer code changes needed                                      │
│  - Messages automatically distributed to consumers                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Durability

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Message Persistence                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌──────────────────────────┐                        │
│  │ Producer │────►│ Message Queue (Durable)  │                        │
│  └──────────┘     ├──────────────────────────┤                        │
│                   │  Memory                   │                        │
│                   │  ┌──────────────────┐    │                        │
│                   │  │ [msg1] [msg2]    │    │                        │
│                   │  └────────┬─────────┘    │                        │
│                   │           │              │                        │
│                   │           ▼              │                        │
│                   │  Disk (WAL)              │                        │
│                   │  ┌──────────────────┐    │                        │
│                   │  │ msg1, msg2, ...  │    │                        │
│                   │  └──────────────────┘    │                        │
│                   └──────────────────────────┘                        │
│                                                                         │
│  Failure recovery:                                                      │
│  1. Broker restarts                                                     │
│  2. Restore messages from disk                                          │
│  3. Resend to consumers                                                 │
│  → Prevent message loss                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Queue vs Topic

### Point-to-Point (Queue)

One message is processed by only one consumer.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Point-to-Point (Queue)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         ┌───────────────────┐                          │
│  ┌──────────┐          │      Queue        │                          │
│  │Producer 1│─────────►│ ┌───┬───┬───┬───┐ │                          │
│  └──────────┘          │ │ A │ B │ C │ D │ │                          │
│                        │ └───┴───┴───┴───┘ │                          │
│  ┌──────────┐          │                   │          ┌──────────┐    │
│  │Producer 2│─────────►│ Each message      │─────────►│Consumer 1│    │
│  └──────────┘          │ processed by only │   A,C    └──────────┘    │
│                        │ one consumer      │                          │
│                        │                   │          ┌──────────┐    │
│                        │                   │─────────►│Consumer 2│    │
│                        │                   │   B,D    └──────────┘    │
│                        └───────────────────┘                          │
│                                                                         │
│  Use cases:                                                             │
│  - Work distribution (Work Queue)                                       │
│  - Order processing                                                     │
│  - Email sending queue                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Publish/Subscribe (Topic)

One message is delivered to all subscribers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Publish/Subscribe (Topic)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         ┌───────────────────┐                          │
│  ┌──────────┐          │      Topic        │         ┌────────────┐   │
│  │Publisher │─────────►│ ┌───┬───┬───┬───┐ │────────►│Subscriber 1│   │
│  └──────────┘          │ │ A │ B │ C │ D │ │  A,B,   │ (all msgs) │   │
│                        │ └───┴───┴───┴───┘ │  C,D    └────────────┘   │
│                        │                   │                          │
│                        │ All subscribers   │         ┌────────────┐   │
│                        │ receive all       │────────►│Subscriber 2│   │
│                        │ messages          │  A,B,   │ (all msgs) │   │
│                        │                   │  C,D    └────────────┘   │
│                        │                   │                          │
│                        │                   │         ┌────────────┐   │
│                        │                   │────────►│Subscriber 3│   │
│                        │                   │  A,B,   │ (all msgs) │   │
│                        └───────────────────┘  C,D    └────────────┘   │
│                                                                         │
│  Use cases:                                                             │
│  - Event broadcast                                                      │
│  - Log collection (multiple systems receive same logs)                  │
│  - Price updates (propagate to all clients)                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Consumer Group (Hybrid)

Kafka style: Topic + distributed processing within group

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Consumer Group                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         ┌───────────────────┐                          │
│                        │      Topic        │                          │
│  ┌──────────┐          │ ┌───┬───┬───┬───┐ │                          │
│  │Publisher │─────────►│ │ A │ B │ C │ D │ │                          │
│  └──────────┘          │ └───┴───┴───┴───┘ │                          │
│                        └─────────┬─────────┘                          │
│                                  │                                     │
│              ┌───────────────────┼───────────────────┐                 │
│              ▼                   ▼                   ▼                 │
│  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐    │
│  │  Consumer Group 1 │ │  Consumer Group 2 │ │  Consumer Group 3 │    │
│  │ (Order Service)   │ │ (Analytics)       │ │ (Notification)    │    │
│  ├───────────────────┤ ├───────────────────┤ ├───────────────────┤    │
│  │ ┌────┐ ┌────┐    │ │ ┌────┐ ┌────┐    │ │ ┌────┐            │    │
│  │ │C1-1│ │C1-2│    │ │ │C2-1│ │C2-2│    │ │ │C3-1│            │    │
│  │ │A,C │ │B,D │    │ │ │A,C │ │B,D │    │ │ │A,B,C,D│         │    │
│  │ └────┘ └────┘    │ │ └────┘ └────┘    │ │ └────┘            │    │
│  └───────────────────┘ └───────────────────┘ └───────────────────┘    │
│                                                                         │
│  Behavior:                                                              │
│  - Each group receives all messages (Pub/Sub)                           │
│  - Within group, messages are distributed (Point-to-Point)              │
│  - Scaling: Add consumers within group                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pattern Comparison

| Characteristic | Queue (P2P) | Topic (Pub/Sub) | Consumer Group |
|------|-------------|-----------------|----------------|
| Message copies | 1:1 | 1:N | 1 per group |
| Load balancing | O | X | O within group |
| Broadcast | X | O | O between groups |
| Scalability | Add consumers | Limited | Flexible |

---

## 4. Message Delivery Guarantees

### At-Most-Once

Messages may be lost, but no duplicates.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     At-Most-Once                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌─────────┐     ┌──────────┐                        │
│  │ Producer │     │  Queue  │     │ Consumer │                        │
│  └────┬─────┘     └────┬────┘     └────┬─────┘                        │
│       │                │               │                               │
│       │── Send(msg) ──►│               │                               │
│       │                │── Deliver ───►│                               │
│       │                │               X Failure before processing!    │
│       │                │               │                               │
│       │                │ (delete msg)  │                               │
│       │                │               │                               │
│       │                │ No resend     │                               │
│       │                │               │                               │
│                                                                         │
│  Implementation: Delete message before ACK                              │
│                                                                         │
│  Advantages:                                                            │
│  - No duplicate processing needed                                       │
│  - Fastest performance                                                  │
│                                                                         │
│  Disadvantages:                                                         │
│  - Possible message loss                                                │
│                                                                         │
│  Use cases:                                                             │
│  - Real-time sensor data (some loss acceptable)                         │
│  - Log collection (performance over completeness)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### At-Least-Once

Messages may be duplicated, but no loss.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     At-Least-Once                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌─────────┐     ┌──────────┐                        │
│  │ Producer │     │  Queue  │     │ Consumer │                        │
│  └────┬─────┘     └────┬────┘     └────┬─────┘                        │
│       │                │               │                               │
│       │── Send(msg) ──►│               │                               │
│       │                │── Deliver ───►│                               │
│       │                │               │ Processing complete           │
│       │                │               X Failure before ACK!           │
│       │                │               │                               │
│       │                │ (no ACK recv) │                               │
│       │                │               │                               │
│       │                │── Redeliver ─►│ ← Resend (duplicate!)         │
│       │                │◄── ACK ───────│                               │
│       │                │               │                               │
│                                                                         │
│  Implementation: Delete message only after ACK                          │
│                                                                         │
│  Advantages:                                                            │
│  - No message loss                                                      │
│                                                                         │
│  Disadvantages:                                                         │
│  - Need duplicate processing (requires idempotency)                     │
│                                                                         │
│  Use cases:                                                             │
│  - Payment processing (no loss allowed, implement idempotency)          │
│  - Order processing                                                     │
│  - Email sending                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Exactly-Once

Theoretically most ideal, but complex to implement.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Exactly-Once                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Pure Exactly-Once is impossible in distributed systems!                │
│  (Two Generals Problem)                                                │
│                                                                         │
│  Practical implementation: "Effectively Once"                           │
│                                                                         │
│  Method 1: At-Least-Once + Idempotent Consumer                          │
│  ┌──────────┐     ┌─────────┐     ┌──────────────────┐                │
│  │ Producer │────►│  Queue  │────►│ Idempotent       │                │
│  └──────────┘     └─────────┘     │ Consumer         │                │
│                                   │ ┌──────────────┐ │                │
│                                   │ │Processed IDs │ │                │
│                                   │ │{id1, id2, ..}│ │                │
│                                   │ └──────────────┘ │                │
│                                   │ Ignore already   │                │
│                                   │ processed        │                │
│                                   └──────────────────┘                │
│                                                                         │
│  Method 2: Transaction-based (Kafka Transactions)                       │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Transaction                                             │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │          │
│  │  │ Read from   │──│ Process     │──│ Write to    │      │          │
│  │  │ input topic │  │             │  │ output topic│      │          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │          │
│  │                                                          │          │
│  │  → All succeed or all fail                               │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                         │
│  Use cases:                                                             │
│  - Financial transactions                                               │
│  - Inventory management                                                 │
│  - Stream processing (Kafka Streams)                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Delivery Guarantee Comparison

| Guarantee Level | Loss | Duplicate | Performance | Complexity |
|----------|------|------|------|--------|
| At-Most-Once | Possible | None | Highest | Low |
| At-Least-Once | None | Possible | Good | Medium |
| Exactly-Once | None | None | Low | High |

---

## 5. Idempotency

### What is Idempotency?

The property that performing the same operation multiple times produces the same result.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Idempotency                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Idempotent operation:                                                  │
│  f(f(x)) = f(x)                                                        │
│                                                                         │
│  Examples:                                                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Operation                │ Idempotent? │ Description         │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │  x = 5                    │  O     │ x=5 no matter how many  │    │
│  │  DELETE /users/123        │  O     │ Deleting already deleted│    │
│  │  PUT /users/123 {name:A}  │  O     │ Overwrite with same val │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │  x = x + 1                │  X     │ Increases each time     │    │
│  │  POST /orders             │  X     │ Creates new order each  │    │
│  │  account.balance -= 100   │  X     │ Deducts each time       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Idempotency Implementation Patterns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Idempotency Implementation Methods                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Method 1: Idempotency Key                                             │
│                                                                         │
│  ┌──────────┐                         ┌────────────────────────┐       │
│  │  Client  │                         │        Server          │       │
│  └────┬─────┘                         │  ┌──────────────────┐  │       │
│       │                               │  │ Processed Keys   │  │       │
│       │──POST /payment                │  │ {key1, key2, ...}│  │       │
│       │  Idempotency-Key: abc123     │  └──────────────────┘  │       │
│       │  {amount: 100}               │                        │       │
│       │                               │  1. Check: abc123?    │       │
│       │                               │  2. Not found → process│       │
│       │                               │  3. Store abc123      │       │
│       │◄─ 200 OK ─────────────────────│                        │       │
│       │                               │                        │       │
│       │──POST /payment (retry)        │                        │       │
│       │  Idempotency-Key: abc123     │                        │       │
│       │  {amount: 100}               │                        │       │
│       │                               │  1. Check: abc123?    │       │
│       │                               │  2. Found → return old │       │
│       │◄─ 200 OK (same response) ─────│                        │       │
│       │                               │                        │       │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Method 2: Version/Conditional Update                                  │
│                                                                         │
│  -- Non-idempotent UPDATE                                               │
│  UPDATE accounts SET balance = balance - 100 WHERE id = 1;             │
│  -- Deducts 100 each execution                                          │
│                                                                         │
│  -- Idempotent UPDATE (using version)                                  │
│  UPDATE accounts                                                        │
│  SET balance = balance - 100, version = 2                               │
│  WHERE id = 1 AND version = 1;                                         │
│  -- Only executes when version = 1 (succeeds only once)                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Method 3: Absolute Value Setting                                      │
│                                                                         │
│  -- Non-idempotent                                                      │
│  INSERT INTO orders (user_id, amount) VALUES (1, 100);                 │
│                                                                         │
│  -- Idempotent (UPSERT)                                                │
│  INSERT INTO orders (order_id, user_id, amount)                        │
│  VALUES ('order-abc123', 1, 100)                                       │
│  ON CONFLICT (order_id) DO NOTHING;                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Idempotent Consumer Implementation Example

```python
class IdempotentConsumer:
    def __init__(self, redis_client, db):
        self.redis = redis_client
        self.db = db

    def process_message(self, message):
        message_id = message['id']

        # 1. Check if already processed
        if self.is_processed(message_id):
            print(f"Message {message_id} already processed, skipping")
            return

        # 2. Process business logic
        try:
            self.handle_business_logic(message)

            # 3. Mark as processed (atomically)
            self.mark_processed(message_id)

        except Exception as e:
            # Can retry on processing failure
            raise

    def is_processed(self, message_id):
        # Use Redis SET (manage memory with TTL)
        return self.redis.sismember("processed_messages", message_id)

    def mark_processed(self, message_id):
        # Auto-delete after 24 hours
        self.redis.sadd("processed_messages", message_id)
        self.redis.expire("processed_messages", 86400)

    def handle_business_logic(self, message):
        # Actual business logic
        if message['type'] == 'payment':
            self.process_payment(message['data'])
```

---

## 6. Message Queue Patterns

### Work Queue (Task Queue)

Distribute tasks to multiple workers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Work Queue Pattern                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                       ┌─────────────────┐                              │
│                       │   Work Queue    │                              │
│  ┌──────────┐        │ ┌───┬───┬───┐   │         ┌──────────┐         │
│  │ Producer │───────►│ │T1 │T2 │T3 │   │────────►│ Worker 1 │         │
│  │  (Web)   │        │ ├───┼───┼───┤   │         └──────────┘         │
│  └──────────┘        │ │T4 │T5 │T6 │   │                              │
│                       │ └───┴───┴───┘   │         ┌──────────┐         │
│                       └─────────────────┘────────►│ Worker 2 │         │
│                                                   └──────────┘         │
│  Use cases:                                        ┌──────────┐         │
│  - Image resizing                           ─────►│ Worker 3 │         │
│  - PDF generation                                  └──────────┘         │
│  - Email sending                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Request-Reply

Asynchronous request-response pattern.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Request-Reply Pattern                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌─────────────┐     ┌──────────┐                    │
│  │ Requester│     │Request Queue│     │ Replier  │                    │
│  │          │     └─────────────┘     │          │                    │
│  │          │     ┌─────────────┐     │          │                    │
│  │          │     │Reply Queue  │     │          │                    │
│  └────┬─────┘     └──────┬──────┘     └────┬─────┘                    │
│       │                  │                 │                           │
│       │─Request(replyTo:Q1,correlationId:C1)─►│                        │
│       │                  │                 │                           │
│       │                  │                 │ Processing                │
│       │                  │                 │                           │
│       │◄──Reply(correlationId:C1)──────────│                           │
│       │                  │                 │                           │
│                                                                         │
│  Message structure:                                                     │
│  Request: {                                                             │
│    correlationId: "req-123",                                           │
│    replyTo: "reply-queue-A",                                           │
│    body: { ... }                                                       │
│  }                                                                      │
│                                                                         │
│  Reply: {                                                               │
│    correlationId: "req-123",  // Match request and reply               │
│    body: { ... }                                                       │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dead Letter Queue (DLQ)

Store failed messages.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Dead Letter Queue                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                       ┌─────────────────┐                              │
│  ┌──────────┐        │   Main Queue    │         ┌──────────┐         │
│  │ Producer │───────►│ ┌───┬───┬───┐   │────────►│ Consumer │         │
│  └──────────┘        │ │ A │ B │ C │   │         └────┬─────┘         │
│                       │ └───┴───┴───┘   │              │                │
│                       └────────┬────────┘              │                │
│                                │                       │                │
│                                │ Processing failed     │                │
│                                │ (retry exceeded)      │                │
│                                ▼                       ▼                │
│                       ┌─────────────────┐        Failed! (msg B)        │
│                       │ Dead Letter Q   │              │                │
│                       │ ┌───┐           │◄─────────────┘                │
│                       │ │ B │           │                               │
│                       │ └───┘           │                               │
│                       └────────┬────────┘                               │
│                                │                                        │
│                                ▼                                        │
│                       Manual review / reprocess                         │
│                                                                         │
│  DLQ usage:                                                             │
│  - Analyze failure cause                                                │
│  - Manual reprocessing                                                  │
│  - Alerting/monitoring                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Priority Queue

Process messages according to priority.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Priority Queue                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌─────────────────────┐     ┌──────────┐            │
│  │ Producer │     │   Priority Queue    │     │ Consumer │            │
│  └────┬─────┘     │                     │     └────┬─────┘            │
│       │           │ High   ┌───┬───┐    │          │                  │
│       │──(P:High)─│───────►│ A │ D │    │──────────│                  │
│       │           │        └───┴───┘    │          │                  │
│       │──(P:Med)──│ Medium ┌───┬───┐    │          │                  │
│       │           │───────►│ B │   │    │          │                  │
│       │           │        └───┴───┘    │          │                  │
│       │──(P:Low)──│ Low    ┌───┬───┐    │          │                  │
│       │           │───────►│ C │ E │    │          │                  │
│       │           │        └───┴───┘    │          │                  │
│       │           └─────────────────────┘          │                  │
│       │                                            │                  │
│       │           Processing order: A → D → B → C → E │                  │
│                                                                         │
│  Implementation methods:                                                │
│  1. Separate queues per priority + weighted polling                    │
│  2. Single queue + heap sort                                            │
│  3. RabbitMQ: x-max-priority setting                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Exercise 1: Choose Communication Method

Select appropriate synchronous/asynchronous method for the following scenarios:

1. User login authentication
2. Send confirmation email after order
3. Payment approval request
4. Log collection
5. Real-time chat message

### Exercise 2: Choose Delivery Guarantee

Select appropriate delivery guarantee level for the following use cases:

1. IoT sensor temperature data
2. Bank transfer request
3. News feed update
4. Order creation event
5. Game player position update

### Exercise 3: Design Idempotency

Design how to make the following operations idempotent:

1. Withdraw 100 from account
2. Decrease product inventory by 1
3. Send email
4. Add points

---

**Previous**: [Data Consistency Patterns](./10_Data_Consistency_Patterns.md) | **Next**: [Message System Comparison](./12_Message_System_Comparison.md)

---

## References

- "Enterprise Integration Patterns" - Gregor Hohpe, Bobby Woolf
- RabbitMQ Official Documentation
- Apache Kafka Documentation
- AWS Messaging Services
- "Designing Data-Intensive Applications" - Martin Kleppmann
