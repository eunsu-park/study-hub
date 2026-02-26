# Message Systems Comparison

**Previous**: [Message Queue Fundamentals](./11_Message_Queue_Basics.md) | **Next**: [Microservices Basics](./13_Microservices_Basics.md)

Difficulty: ⭐⭐⭐

## Learning Objectives

1. Explain the architecture and core abstractions of Apache Kafka (topics, partitions, consumer groups, log-based storage)
2. Describe RabbitMQ's AMQP model including exchanges, bindings, queues, and acknowledgment mechanisms
3. Compare AWS SQS (standard vs FIFO) and SNS for cloud-native messaging scenarios
4. Evaluate message systems against selection criteria such as throughput, latency, ordering, durability, and operational complexity
5. Design hybrid messaging architectures that combine multiple systems for different workload characteristics
6. Select the appropriate message system for a given set of requirements and constraints

---

Choosing a message system is one of the most consequential infrastructure decisions a team makes -- it affects throughput ceilings, operational burden, and even how you model your data pipelines for years to come. Kafka, RabbitMQ, and cloud-managed services like AWS SQS/SNS each embody fundamentally different design philosophies, and picking the wrong one can mean expensive re-architecture later. This lesson equips you with the mental framework to make that choice confidently.

## Table of Contents

1. [Apache Kafka](#1-apache-kafka)
2. [RabbitMQ](#2-rabbitmq)
3. [AWS SQS/SNS](#3-aws-sqssns)
4. [System Comparison and Selection Criteria](#4-system-comparison-and-selection-criteria)
5. [Hybrid Architecture](#5-hybrid-architecture)
6. [Practice Problems](#6-practice-problems)

---

## 1. Apache Kafka

### Kafka Overview

Kafka is a distributed streaming platform developed by LinkedIn that provides high throughput and durability.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Kafka Architecture                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        Kafka Cluster                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │   Broker 1  │  │   Broker 2  │  │   Broker 3  │               │  │
│  │  │  (Leader)   │  │ (Follower)  │  │ (Follower)  │               │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ▲                                          │
│                              │ ZooKeeper / KRaft                       │
│                              │ (Cluster metadata management)           │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        ZooKeeper Ensemble                         │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐                              │  │
│  │  │  ZK 1  │  │  ZK 2  │  │  ZK 3  │                              │  │
│  │  └────────┘  └────────┘  └────────┘                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Topic and Partition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Topic and Partition                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Topic: "orders" (3 Partitions)                                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Partition 0 (Broker 1)                                         │   │
│  │  ┌────┬────┬────┬────┬────┬────┐                               │   │
│  │  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ ──► offset                    │   │
│  │  └────┴────┴────┴────┴────┴────┘                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Partition 1 (Broker 2)                                         │   │
│  │  ┌────┬────┬────┬────┬────┐                                    │   │
│  │  │ 0  │ 1  │ 2  │ 3  │ 4  │                                    │   │
│  │  └────┴────┴────┴────┴────┘                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Partition 2 (Broker 3)                                         │   │
│  │  ┌────┬────┬────┬────┬────┬────┬────┐                          │   │
│  │  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │                          │   │
│  │  └────┴────┴────┴────┴────┴────┴────┘                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Characteristics:                                                       │
│  - Order is guaranteed within each partition                            │
│  - Order is NOT guaranteed across partitions                            │
│  - Parallel processing scales with partition count                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Consumer Group

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Consumer Group                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Topic: "orders" (4 Partitions)                                        │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │  P0       P1       P2       P3                                │     │
│  └───┬───────┬───────┬───────┬───────────────────────────────────┘     │
│      │       │       │       │                                          │
│      │       │       │       │                                          │
│  ┌───┴───────┴───────┴───────┴───────────────────────────────────┐     │
│  │           Consumer Group A (Order Service)                    │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │     │
│  │  │Consumer 1│  │Consumer 2│  │Consumer 3│                    │     │
│  │  │  P0, P1  │  │    P2    │  │    P3    │                    │     │
│  │  └──────────┘  └──────────┘  └──────────┘                    │     │
│  └───────────────────────────────────────────────────────────────┘     │
│      │       │       │       │                                          │
│  ┌───┴───────┴───────┴───────┴───────────────────────────────────┐     │
│  │           Consumer Group B (Analytics)                        │     │
│  │  ┌──────────────────────────────────────────────┐            │     │
│  │  │            Consumer 1 (P0, P1, P2, P3)       │            │     │
│  │  └──────────────────────────────────────────────┘            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Rules:                                                                 │
│  - One partition is consumed by only one Consumer in the group          │
│  - Consumer > Partition count → Some Consumers are idle                 │
│  - Consumer < Partition count → Some Consumers handle multiple          │
│    partitions                                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Offset Management

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Offset Management                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Partition 0:                                                          │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐                 │
│  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │                 │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘                 │
│                 ▲              ▲                   ▲                   │
│                 │              │                   │                   │
│       Committed Offset   Current Position    Log End Offset            │
│        (Position the      (Currently         (Latest message)          │
│         Consumer           processing)                                 │
│         acknowledged)                                                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  __consumer_offsets (internal topic):                                   │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Group: order-service, Topic: orders, Partition: 0             │    │
│  │  Committed Offset: 5                                           │    │
│  │  Timestamp: 2024-01-15T10:30:00Z                               │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Offset Reset Policies:                                                 │
│  - earliest: From the beginning (all messages)                          │
│  - latest: From the newest (new messages only)                          │
│  - none: Error if no offset exists                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Kafka as a Distributed Log

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Kafka = Distributed Commit Log                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Traditional Message Queue:                                             │
│  ┌─────────────────────────────────────┐                               │
│  │  [msg] ─→ Queue ─→ Consumer ─→ Delete│                               │
│  └─────────────────────────────────────┘                               │
│                                                                         │
│  Kafka (Log-based):                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Append-Only Log                                                │   │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┐                     │   │
│  │  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ ──► Append new msg  │   │
│  │  └────┴────┴────┴────┴────┴────┴────┴────┘                     │   │
│  │       ▲         ▲              ▲                                │   │
│  │  Consumer A  Consumer B   Consumer C                            │   │
│  │  (offset 1)  (offset 3)   (offset 6)                           │   │
│  │                                                                 │   │
│  │  - Messages are NOT deleted (retained until retention period)   │   │
│  │  - Each Consumer manages its own offset                         │   │
│  │  - Can reprocess past messages (replay)                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Advantages:                                                            │
│  - Supports event sourcing                                              │
│  - Easy reprocessing on failure recovery                                │
│  - Multiple Consumers process same data independently                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Kafka Replication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Kafka Replication                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Topic: orders, Partition 0 (Replication Factor: 3)                    │
│                                                                         │
│  ┌────────────────────┐                                                │
│  │     Broker 1       │                                                │
│  │  ┌──────────────┐  │                                                │
│  │  │ P0 (Leader)  │◄─┼────── Producer (writes)                        │
│  │  │ [0][1][2][3] │──┼────── Consumer (reads)                         │
│  │  └──────────────┘  │                                                │
│  └─────────┬──────────┘                                                │
│            │ Replicate                                                  │
│     ┌──────┴──────┐                                                    │
│     ▼             ▼                                                    │
│  ┌──────────────────┐  ┌──────────────────┐                           │
│  │    Broker 2      │  │    Broker 3      │                           │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                           │
│  │ │P0 (Follower) │ │  │ │P0 (Follower) │ │                           │
│  │ │ [0][1][2][3] │ │  │ │ [0][1][2][3] │ │                           │
│  │ └──────────────┘ │  │ └──────────────┘ │                           │
│  └──────────────────┘  └──────────────────┘                           │
│                                                                         │
│  ISR (In-Sync Replicas):                                               │
│  - List of replicas synchronized with Leader                            │
│  - acks=all: ACK after write complete to all ISR                        │
│  - On Leader failure, one of ISR becomes new Leader                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. RabbitMQ

### RabbitMQ Overview

RabbitMQ is a traditional message broker based on the AMQP protocol.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RabbitMQ Architecture                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌───────────────────────────────────┐    ┌─────────┐ │
│  │ Producer │────►│          RabbitMQ Broker          │───►│Consumer │ │
│  └──────────┘     │  ┌──────────┐    ┌─────────────┐  │    └─────────┘ │
│                   │  │ Exchange │───►│    Queue    │  │                │
│                   │  └──────────┘    └─────────────┘  │                │
│                   └───────────────────────────────────┘                │
│                                                                         │
│  Core Components:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. Exchange: Defines message routing rules                      │   │
│  │  2. Queue: Message storage                                       │   │
│  │  3. Binding: Connection between Exchange and Queue               │   │
│  │  4. Routing Key: Key for routing decisions                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Exchange Types

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Exchange Types                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Direct Exchange:                                                   │
│  ┌──────────┐     ┌──────────┐     ┌─────────┐                        │
│  │ Producer │     │  Direct  │     │ Queue A │ (routing_key=error)    │
│  │          │────►│ Exchange │────►│         │                        │
│  │ (error)  │     │          │     └─────────┘                        │
│  └──────────┘     │          │     ┌─────────┐                        │
│                   │          │────►│ Queue B │ (routing_key=info)     │
│                   └──────────┘     └─────────┘                        │
│                                                                         │
│  → Delivers to Queue where routing_key exactly matches                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  2. Topic Exchange:                                                    │
│  ┌──────────┐     ┌──────────┐     ┌─────────────────────────────────┐│
│  │ Producer │     │  Topic   │     │ Queue A: *.error                ││
│  │          │────►│ Exchange │────►│ (orders.error, users.error)     ││
│  │(orders.  │     │          │     └─────────────────────────────────┘│
│  │ error)   │     │          │     ┌─────────────────────────────────┐│
│  └──────────┘     │          │────►│ Queue B: orders.*               ││
│                   │          │     │ (orders.error, orders.info)     ││
│                   └──────────┘     └─────────────────────────────────┘│
│                                                                         │
│  → Pattern matching (* = one word, # = zero or more)                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  3. Fanout Exchange:                                                   │
│  ┌──────────┐     ┌──────────┐     ┌─────────┐                        │
│  │ Producer │     │  Fanout  │────►│ Queue A │                        │
│  │          │────►│ Exchange │────►│ Queue B │  Copy to all Queues    │
│  │          │     │          │────►│ Queue C │                        │
│  └──────────┘     └──────────┘     └─────────┘                        │
│                                                                         │
│  → Ignores routing_key, broadcasts to all bound Queues                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  4. Headers Exchange:                                                  │
│  ┌──────────┐     ┌──────────┐     ┌─────────┐                        │
│  │ Producer │     │ Headers  │────►│ Queue A │                        │
│  │ headers: │────►│ Exchange │     │ x-match:│                        │
│  │{type:pdf}│     │          │     │ all,    │                        │
│  └──────────┘     └──────────┘     │type:pdf │                        │
│                                    └─────────┘                        │
│                                                                         │
│  → Routing based on message headers                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### RabbitMQ Routing Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Topic Exchange Routing Example                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Topic Exchange: "logs"                                                │
│                                                                         │
│  Bindings:                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Queue          Binding Key                                     │   │
│  │  error_logs     *.error                                         │   │
│  │  web_logs       web.*                                           │   │
│  │  all_logs       #                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Message Routing:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Routing Key      Goes to                                       │   │
│  │  web.error        error_logs, web_logs, all_logs               │   │
│  │  web.info         web_logs, all_logs                           │   │
│  │  db.error         error_logs, all_logs                         │   │
│  │  cache.warning    all_logs                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Pattern Rules:                                                         │
│  - * (star): Exactly one word                                           │
│  - # (hash): Zero or more words                                         │
│  - Example: stock.# → stock.usd.nyse, stock.eur                         │
│             stock.* → stock.usd (O), stock.usd.nyse (X)                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### RabbitMQ Message Acknowledgment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Message Acknowledgment                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌─────────┐     ┌──────────┐                        │
│  │  Queue   │     │Consumer │     │   App    │                        │
│  └────┬─────┘     └────┬────┘     └────┬─────┘                        │
│       │                │               │                               │
│       │── Deliver ────►│               │                               │
│       │                │── Process ───►│                               │
│       │                │               │ Processing...                 │
│       │                │◄── Done ──────│                               │
│       │◄── ACK ────────│               │                               │
│       │                │               │                               │
│       │ (Delete msg)   │               │                               │
│                                                                         │
│  ACK Modes:                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  auto-ack: true   → ACK immediately on delivery (At-Most-Once) │   │
│  │  auto-ack: false  → Explicit ACK required (At-Least-Once)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  NACK/Reject:                                                          │
│  - basic_nack: Reject message (with requeue option)                     │
│  - basic_reject: Reject single message                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### RabbitMQ High Availability

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RabbitMQ Clustering                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Classic Mirrored Queue (deprecated):                                  │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │  Node 1 (Master)    Node 2 (Mirror)    Node 3 (Mirror)        │     │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │     │
│  │  │ Queue: Q1   │───►│ Queue: Q1   │───►│ Queue: Q1   │        │     │
│  │  │ [msg][msg]  │    │ [msg][msg]  │    │ [msg][msg]  │        │     │
│  │  └─────────────┘    └─────────────┘    └─────────────┘        │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Quorum Queue (Recommended):                                            │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │  Raft-based Consensus                                         │     │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │     │
│  │  │   Leader    │    │  Follower   │    │  Follower   │        │     │
│  │  │    Node 1   │◄──►│    Node 2   │◄──►│    Node 3   │        │     │
│  │  └─────────────┘    └─────────────┘    └─────────────┘        │     │
│  │                                                                │     │
│  │  - Stronger data consistency                                   │     │
│  │  - Automatic leader election                                   │     │
│  │  - Improved poison message handling                            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. AWS SQS/SNS

### AWS SQS (Simple Queue Service)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AWS SQS Architecture                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌─────────────────────────────────┐     ┌─────────┐ │
│  │ Producer │────►│           SQS Queue             │────►│Consumer │ │
│  └──────────┘     │  ┌─────────────────────────┐    │     └─────────┘ │
│                   │  │  Distributed Storage    │    │                 │
│                   │  │  (Multiple AZs)         │    │                 │
│                   │  │  [msg][msg][msg][msg]   │    │                 │
│                   │  └─────────────────────────┘    │                 │
│                   └─────────────────────────────────┘                 │
│                                                                         │
│  Fully Managed Service:                                                 │
│  - Unlimited throughput                                                 │
│  - Auto scaling                                                         │
│  - No infrastructure management                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Standard vs FIFO Queue

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Standard Queue vs FIFO Queue                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Standard Queue:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │  Sent: A → B → C → D                                            │   │
│  │  Received: A → C → B → D (Order not guaranteed)                 │   │
│  │            or A → A → B → C → D (Duplicates possible)           │   │
│  │                                                                 │   │
│  │  Characteristics:                                               │   │
│  │  - Max throughput: Nearly unlimited                             │   │
│  │  - At-Least-Once delivery                                       │   │
│  │  - Best-Effort ordering                                         │   │
│  │  - Price: Low                                                   │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  FIFO Queue:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │  Sent: A → B → C → D                                            │   │
│  │  Received: A → B → C → D (Order guaranteed!)                    │   │
│  │                                                                 │   │
│  │  Characteristics:                                               │   │
│  │  - Max throughput: 3,000 msgs/sec (30,000 with batching)        │   │
│  │  - Exactly-Once processing                                      │   │
│  │  - Strict order guarantee                                       │   │
│  │  - Parallel processing with Message Group ID                    │   │
│  │  - Price: About 1.2x Standard                                   │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### FIFO Message Groups

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FIFO Message Groups                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    FIFO Queue                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  Message Group: user-123                                 │   │   │
│  │  │  [order1] → [order2] → [order3]  Order guaranteed        │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  Message Group: user-456                                 │   │   │
│  │  │  [order1] → [order2]             Order guaranteed        │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  Message Group: user-789                                 │   │   │
│  │  │  [order1]                        Order guaranteed        │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                    │           │           │                           │
│                    ▼           ▼           ▼                           │
│              ┌─────────┐ ┌─────────┐ ┌─────────┐                      │
│              │Consumer1│ │Consumer2│ │Consumer3│                      │
│              └─────────┘ └─────────┘ └─────────┘                      │
│                                                                         │
│  - Order guaranteed only within same Group                              │
│  - Different Groups can be processed in parallel                        │
│  - Balance between scalability and order guarantee                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### AWS SNS (Simple Notification Service)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AWS SNS (Pub/Sub)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                      ┌─────────────────┐                               │
│  ┌──────────┐        │   SNS Topic     │                               │
│  │Publisher │───────►│  "order-events" │                               │
│  └──────────┘        └────────┬────────┘                               │
│                               │                                         │
│         ┌─────────────────────┼─────────────────────┐                  │
│         │                     │                     │                  │
│         ▼                     ▼                     ▼                  │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│  │  SQS Queue  │      │    Lambda   │      │    HTTP     │            │
│  │ (Order Svc) │      │ (Analytics) │      │ (Webhook)   │            │
│  └─────────────┘      └─────────────┘      └─────────────┘            │
│                                                                         │
│  Supported Subscribers:                                                 │
│  - SQS Queue                                                           │
│  - Lambda Function                                                     │
│  - HTTP/HTTPS Endpoint                                                 │
│  - Email/SMS                                                           │
│  - Mobile Push                                                         │
│  - Kinesis Data Firehose                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### SNS + SQS Fan-out Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SNS + SQS Fan-out                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌─────────────┐                                     │
│  │ Publisher│────►│  SNS Topic  │                                     │
│  │ (Order)  │     │             │                                     │
│  └──────────┘     └──────┬──────┘                                     │
│                          │                                             │
│      ┌───────────────────┼───────────────────┐                        │
│      │                   │                   │                        │
│      ▼                   ▼                   ▼                        │
│  ┌────────┐         ┌────────┐         ┌────────┐                     │
│  │SQS: Q1 │         │SQS: Q2 │         │SQS: Q3 │                     │
│  │Inventory│        │Payment │         │Email   │                     │
│  └───┬────┘         └───┬────┘         └───┬────┘                     │
│      │                  │                  │                          │
│      ▼                  ▼                  ▼                          │
│  ┌────────┐         ┌────────┐         ┌────────┐                     │
│  │Inventory│        │Payment │         │Email   │                     │
│  │Service │         │Service │         │Service │                     │
│  └────────┘         └────────┘         └────────┘                     │
│                                                                         │
│  Advantages:                                                            │
│  - Complete decoupling between services                                 │
│  - Independent scaling per service                                      │
│  - Fault isolation                                                      │
│  - Individual retry/DLQ management                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. System Comparison and Selection Criteria

### Core Characteristics Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Message System Comparison                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────┬────────────┬────────────┬────────────┐                │
│  │  Feature   │   Kafka    │  RabbitMQ  │  AWS SQS   │                │
│  ├────────────┼────────────┼────────────┼────────────┤                │
│  │ Throughput │ Very High  │   High     │   High     │                │
│  │            │ Millions/s │ 10K+/sec   │ Unlimited* │                │
│  ├────────────┼────────────┼────────────┼────────────┤                │
│  │ Latency    │   Low      │ Very Low   │  Medium    │                │
│  │            │ ms level   │ sub-ms     │ tens of ms │                │
│  ├────────────┼────────────┼────────────┼────────────┤                │
│  │ Order      │ Within     │ Within     │ FIFO only  │                │
│  │ Guarantee  │ partition  │ queue      │            │                │
│  ├────────────┼────────────┼────────────┼────────────┤                │
│  │ Message    │ Config     │ Delete on  │ Max 14     │                │
│  │ Retention  │ period     │ consume    │ days       │                │
│  ├────────────┼────────────┼────────────┼────────────┤                │
│  │ Replay     │    O       │     X      │     X      │                │
│  ├────────────┼────────────┼────────────┼────────────┤                │
│  │ Routing    │  Simple    │ Flexible   │  Simple    │                │
│  ├────────────┼────────────┼────────────┼────────────┤                │
│  │ Ops        │   High     │  Medium    │   Low      │                │
│  │ Complexity │            │            │            │                │
│  ├────────────┼────────────┼────────────┼────────────┤                │
│  │ Cost Model │ Infra cost │ Infra cost │ Usage-based│                │
│  └────────────┴────────────┴────────────┴────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Use Case Recommendations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Use Case Selection Guide                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Choose Kafka when:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Real-time streaming/analytics (clickstream, logs)            │   │
│  │  - Event sourcing / CQRS                                        │   │
│  │  - High throughput needed                                       │   │
│  │  - Message replay needed                                        │   │
│  │  - Long-term data retention                                     │   │
│  │  Examples: Log collection, metrics collection, real-time        │   │
│  │  analytics                                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Choose RabbitMQ when:                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Complex routing requirements                                  │   │
│  │  - Low latency essential                                         │   │
│  │  - Traditional work queue                                        │   │
│  │  - Various messaging patterns (RPC, Priority Queue)             │   │
│  │  - AMQP protocol needed                                          │   │
│  │  Examples: Work distribution, microservices communication, IoT  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Choose AWS SQS/SNS when:                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Leveraging AWS ecosystem                                      │   │
│  │  - Minimize infrastructure management                            │   │
│  │  - Unpredictable traffic                                         │   │
│  │  - Quick setup needed                                            │   │
│  │  - Lambda triggers                                               │   │
│  │  Examples: Serverless architecture, startups, rapid prototyping │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Selection Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Message System Selection Flow                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    Start                                                │
│                      │                                                  │
│                      ▼                                                  │
│              ┌───────────────┐                                         │
│              │ Using AWS?    │                                         │
│              │ Want minimal  │                                         │
│              │ management?   │                                         │
│              └───────┬───────┘                                         │
│                 Yes  │   No                                            │
│                 ┌────┴────┐                                            │
│                 ▼         ▼                                            │
│           ┌─────────┐ ┌────────────────┐                              │
│           │AWS SQS/ │ │ Event streaming/│                              │
│           │SNS      │ │ Replay needed?  │                              │
│           └─────────┘ └───────┬────────┘                              │
│                          Yes  │   No                                   │
│                          ┌────┴────┐                                   │
│                          ▼         ▼                                   │
│                    ┌─────────┐ ┌────────────────┐                     │
│                    │  Kafka  │ │ Complex routing/│                     │
│                    │         │ │ Low latency?   │                     │
│                    └─────────┘ └───────┬────────┘                     │
│                                   Yes  │   No                          │
│                                   ┌────┴────┐                          │
│                                   ▼         ▼                          │
│                             ┌─────────┐ ┌─────────┐                   │
│                             │RabbitMQ │ │  Either │                   │
│                             │         │ │  works  │                   │
│                             └─────────┘ └─────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Hybrid Architecture

### Kafka + RabbitMQ Combination

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Hybrid Architecture Example                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                          ┌─────────────────────────────────────────┐   │
│                          │            Event Bus (Kafka)            │   │
│  ┌───────────┐          │  - Event store                          │   │
│  │  Service  │─────────►│  - Event replay                         │   │
│  │  Events   │          │  - Analytics pipeline                   │   │
│  └───────────┘          └────────────────┬────────────────────────┘   │
│                                          │                             │
│                                          │ Event transform/filter     │
│                                          ▼                             │
│                          ┌─────────────────────────────────────────┐   │
│                          │           Task Queue (RabbitMQ)         │   │
│                          │  - Work distribution                    │   │
│                          │  - Complex routing                      │   │
│                          │  - Priority handling                    │   │
│                          └────────────────────────────────────────┘   │
│                                    │                                    │
│                          ┌─────────┼─────────┐                        │
│                          ▼         ▼         ▼                        │
│                    ┌─────────┐┌─────────┐┌─────────┐                  │
│                    │Worker 1 ││Worker 2 ││Worker 3 │                  │
│                    └─────────┘└─────────┘└─────────┘                  │
│                                                                         │
│  Leveraging each system's strengths:                                    │
│  - Kafka: Event log, streaming, analytics                               │
│  - RabbitMQ: Work queue, complex workflows                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Practice Problems

### Practice 1: System Selection

Choose the appropriate message system for the following requirements and explain your reasoning:

1. Real-time stock trading system (millisecond latency)
2. E-commerce order event log (3-year retention)
3. Image resizing work queue (priority needed)
4. Serverless-based startup MVP
5. IoT sensor data collection (1 million events/second)

### Practice 2: Kafka Topic Design

Design Kafka topics for an online shopping mall order system:
- Partition count decision
- Partition key selection
- Consumer Group configuration
- Retention policy

### Practice 3: RabbitMQ Routing Design

Design RabbitMQ Exchange/Queue for a log collection system:
- Routing by log level (error, warn, info, debug)
- Routing by service (web, api, db)
- All error logs to notification service
- All logs from specific service to debugging service

---

**Previous**: [Message Queue Fundamentals](./11_Message_Queue_Basics.md) | **Next**: [Microservices Basics](./13_Microservices_Basics.md)

---

## References

- Apache Kafka Documentation
- RabbitMQ Official Documentation
- AWS SQS/SNS Documentation
- "Kafka: The Definitive Guide" - Neha Narkhede
- "RabbitMQ in Depth" - Gavin M. Roy
