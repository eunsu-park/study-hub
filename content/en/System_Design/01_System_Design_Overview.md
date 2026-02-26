# System Design Overview

**Next**: [Scalability Basics](./02_Scalability_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what system design is and why it matters for building large-scale software systems
2. Describe the evaluation criteria used in system design interviews
3. Apply a structured four-step framework to approach any system design problem
4. Perform back-of-the-envelope calculations for estimating capacity, storage, and bandwidth
5. Recall commonly used performance numbers (latency, throughput, storage) for quick estimation
6. Sketch a high-level architecture diagram for a simple system given a set of requirements

**Difficulty**: ⭐
**Estimated Study Time**: 2 hours
**Prerequisites**: Programming basics, basic web service concepts

---

Every application you use daily -- from social media feeds to ride-sharing services -- relies on carefully designed systems that handle millions of requests per second without breaking a sweat. System design is the discipline that makes this possible, and it is also one of the most heavily tested skills in senior engineering interviews. This lesson gives you the foundational vocabulary, estimation toolkit, and structured thinking framework you will use throughout the rest of the course.

## Table of Contents

1. [What is System Design?](#1-what-is-system-design)
2. [Interview Evaluation Criteria](#2-interview-evaluation-criteria)
3. [Problem Approach Framework](#3-problem-approach-framework)
4. [Back-of-the-envelope Calculations](#4-back-of-the-envelope-calculations)
5. [Commonly Used Numbers](#5-commonly-used-numbers)
6. [Practice Problems](#6-practice-problems)
7. [References](#7-references)

---

## 1. What is System Design?

### 1.1 Definition

System design is the process of defining the architecture of complex software systems. The goal is to analyze requirements, design components, and create systems with scalability and reliability.

```
┌─────────────────────────────────────────────────────────────────┐
│                    What is System Design?                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Requirements   →   Architecture   →   Implementation           │
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│  │Functional│     │Component │     │Code      │                │
│  │Req.      │ ──▶ │Design    │ ──▶ │Impl.     │                │
│  │          │     │          │     │          │                │
│  │Non-Func. │     │Data      │     │Testing   │                │
│  │Req.      │     │Flow      │     │          │                │
│  └──────────┘     └──────────┘     └──────────┘                │
│                                                                  │
│  System Design = Requirements → Architecture Decisions          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Why System Design is Important

```
┌─────────────────────────────────────────────────────────────────┐
│               Importance of System Design                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Scalability                                                 │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ System must handle growth from 100 to                  │  │
│     │ 1,000,000 users                                        │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  2. Reliability                                                 │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ Service must continue even with server failures        │  │
│     │ No data loss                                           │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  3. Maintainability                                             │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ Easy to add new features                               │  │
│     │ Bug fixes shouldn't affect other parts                 │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  4. Performance                                                 │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ Fast response times                                    │  │
│     │ Sufficient throughput                                  │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 System Design vs Coding Interview

| Category | Coding Interview | System Design Interview |
|------|----------|-----------------|
| Purpose | Evaluate algorithm skills | Evaluate architecture design skills |
| Answer | Clear correct answer exists | Multiple answers possible (trade-offs) |
| Format | Code writing | Whiteboard/diagrams |
| Evaluation | Correctness, efficiency | Thought process, communication |
| Level | Junior~Senior | Mainly Senior+ |

---

## 2. Interview Evaluation Criteria

### 2.1 Key Evaluation Areas

```
┌─────────────────────────────────────────────────────────────────┐
│                 Interview Evaluation Criteria                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  1. Problem Scoping                                        │ │
│  │     • Do you clarify requirements?                         │ │
│  │     • Do you ask appropriate questions?                    │ │
│  │     • Do you state assumptions?                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  2. High-Level Design                                      │ │
│  │     • Do you identify main components?                     │ │
│  │     • Is data flow clear?                                  │ │
│  │     • Do you design APIs appropriately?                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  3. Deep Dive                                              │ │
│  │     • Do you thoroughly cover core components?             │ │
│  │     • Is data model appropriate?                           │ │
│  │     • Do you identify potential bottlenecks?               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  4. Trade-offs                                             │ │
│  │     • Do you know pros/cons of options?                    │ │
│  │     • Do you explain why you chose specific tech?          │ │
│  │     • Do you consider constraints?                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Expectations by Level

```
┌─────────────────────────────────────────────────────────────────┐
│              Expectations by Experience Level                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Junior (0-2 years)                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Understand basic components (web server, DB, cache)      │ │
│  │ • Explain data flow of simple systems                      │ │
│  │ • Basic API design                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Mid-level (2-5 years)                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Design scalable architecture                             │ │
│  │ • Apply caching, load balancing                            │ │
│  │ • Database selection and schema design                     │ │
│  │ • Basic trade-offs discussion                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Senior (5+ years)                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Design large-scale distributed systems                   │ │
│  │ • Analyze complex trade-offs                               │ │
│  │ • Consider failure recovery, security                      │ │
│  │ • Cost optimization                                        │ │
│  │ • Microservices architecture                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Problem Approach Framework

### 3.1 4-Step Approach

```
┌─────────────────────────────────────────────────────────────────┐
│            System Design 4-Step Framework                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Clarify Requirements (5 min)                      │   │
│  │                                                           │   │
│  │  "Design a Twitter-like service"                          │   │
│  │                                                           │   │
│  │  Questions to ask:                                        │   │
│  │  • Core features? (tweets, timeline, follow?)             │   │
│  │  • User scale? (DAU 1M? 100M?)                            │   │
│  │  • Read/write ratio? (typically 100:1)                    │   │
│  │  • Media support? (images, videos)                        │   │
│  │  • Real-time notifications needed?                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Scale Estimation (5 min)                          │   │
│  │                                                           │   │
│  │  • Calculate QPS (Queries Per Second)                     │   │
│  │  • Estimate storage capacity                              │   │
│  │  • Calculate bandwidth                                    │   │
│  │  • Estimate server count                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STEP 3: High-Level Design (15-20 min)                     │   │
│  │                                                           │   │
│  │  • System architecture diagram                            │   │
│  │  • Main components (client, server, DB, cache, etc.)      │   │
│  │  • Data flow                                              │   │
│  │  • API design                                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STEP 4: Detailed Design (15-20 min)                       │   │
│  │                                                           │   │
│  │  • Database schema                                        │   │
│  │  • Core algorithms/data structures                        │   │
│  │  • Scaling strategies (sharding, replication)             │   │
│  │  • Resolve bottlenecks                                    │   │
│  │  • Failure handling                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Example Requirements Clarification Questions

```
Functional Requirements:
┌─────────────────────────────────────────────────────────────────┐
│ • What are the 3 core features of this system?                  │
│ • What actions can users perform?                               │
│ • Which clients to support: mobile/web/API?                     │
│ • Is authentication/authorization needed?                       │
│ • Is search functionality required?                             │
└─────────────────────────────────────────────────────────────────┘

Non-Functional Requirements:
┌─────────────────────────────────────────────────────────────────┐
│ • Expected user count? (DAU, MAU)                               │
│ • Response time requirements? (p99 < 200ms?)                    │
│ • Availability requirements? (99.9%? 99.99%?)                   │
│ • Data consistency vs availability - which is more important?   │
│ • Global or regional service?                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 High-Level Design Example

```
                      Twitter High-Level Design

    ┌─────────┐     ┌─────────────┐     ┌────────────────┐
    │ Mobile  │     │             │     │                │
    │   App   │────▶│    Load     │────▶│   Web/API      │
    │         │     │  Balancer   │     │   Servers      │
    └─────────┘     │             │     │                │
                    └─────────────┘     └───────┬────────┘
    ┌─────────┐           │                     │
    │   Web   │           │                     │
    │ Browser │───────────┘                     │
    │         │                                 ▼
    └─────────┘                    ┌────────────────────┐
                                   │                    │
                    ┌──────────────┤   Service Layer    │
                    │              │                    │
                    │              └────────────────────┘
                    │                        │
           ┌────────┴────────┐               │
           ▼                 ▼               ▼
    ┌────────────┐    ┌────────────┐  ┌────────────┐
    │   Cache    │    │  Database  │  │   Message  │
    │  (Redis)   │    │  (MySQL)   │  │   Queue    │
    └────────────┘    └────────────┘  └────────────┘
                             │
                             ▼
                      ┌────────────┐
                      │   Object   │
                      │  Storage   │
                      │   (S3)     │
                      └────────────┘
```

---

## 4. Back-of-the-envelope Calculations

### 4.1 QPS (Queries Per Second) Calculation

```
┌─────────────────────────────────────────────────────────────────┐
│                    QPS Calculation Method                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Example: Twitter Tweet Read QPS                                │
│                                                                  │
│  Given:                                                         │
│  • DAU (Daily Active Users): 300 million                        │
│  • Average daily tweet views per user: 100                      │
│                                                                  │
│  Calculation:                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Daily total views = 300,000,000 × 100 = 30,000,000,000    │ │
│  │                                                            │ │
│  │ Average QPS = 30,000,000,000 / 86,400 ≈ 350,000 QPS       │ │
│  │                                                            │ │
│  │ Peak QPS = Average QPS × 2~3 ≈ 700,000 ~ 1,000,000 QPS    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Note: 86,400 = 24 hours × 60 minutes × 60 seconds             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Storage Capacity Calculation

```
┌─────────────────────────────────────────────────────────────────┐
│                Storage Capacity Calculation Method               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Example: Twitter 5-year storage capacity                       │
│                                                                  │
│  Given:                                                         │
│  • DAU: 300 million                                             │
│  • Daily avg tweets: 2 (10% of users post)                      │
│  • Average tweet size: 250 bytes (text only)                    │
│  • Image ratio: 20%, avg image size: 500KB                      │
│                                                                  │
│  Calculation:                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Daily tweets = 300M × 10% × 2 = 60M                        │ │
│  │                                                            │ │
│  │ Text storage:                                              │ │
│  │   Daily = 60M × 250B = 15GB                                │ │
│  │   Yearly = 15GB × 365 = 5.5TB                              │ │
│  │   5 years = 5.5TB × 5 = 27.5TB                             │ │
│  │                                                            │ │
│  │ Image storage:                                             │ │
│  │   Daily = 60M × 20% × 500KB = 6TB                          │ │
│  │   Yearly = 6TB × 365 = 2.2PB                               │ │
│  │   5 years = 2.2PB × 5 = 11PB                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Bandwidth Calculation

```
┌─────────────────────────────────────────────────────────────────┐
│                   Bandwidth Calculation Method                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Example: Video streaming service                               │
│                                                                  │
│  Given:                                                         │
│  • Concurrent viewers: 1 million                                │
│  • Average bitrate: 5 Mbps (1080p standard)                     │
│                                                                  │
│  Calculation:                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Total required bandwidth = 1,000,000 × 5 Mbps = 5,000,000  │ │
│  │                          = 5,000 Gbps = 5 Tbps             │ │
│  │                                                            │ │
│  │ Daily data transfer (assume avg 2hr viewing):              │ │
│  │   = 100M viewers × 5Mbps × 7200sec                         │ │
│  │   = 3.6 × 10^15 bits = 450 TB/day                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Commonly Used Numbers

### 5.1 Powers of 2

| Power | Approximate | Name | Bytes |
|----------|--------|------|--------|
| 2^10 | 1,000 | 1 Thousand | 1 KB |
| 2^20 | 1,000,000 | 1 Million | 1 MB |
| 2^30 | 1,000,000,000 | 1 Billion | 1 GB |
| 2^40 | 1,000,000,000,000 | 1 Trillion | 1 TB |
| 2^50 | - | 1 Quadrillion | 1 PB |

### 5.2 Time Unit Conversion

```
┌─────────────────────────────────────────────────────────────────┐
│                    Time Unit Reference Table                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1 day  = 86,400 seconds   (≈ 100,000 seconds approx.)         │
│  1 week  = 604,800 seconds  (≈ 600,000 seconds approx.)        │
│  1 month  = 2,592,000 seconds (≈ 2.5M seconds approx.)         │
│  1 year  = 31,536,000 seconds (≈ 30M seconds approx.)          │
│                                                                  │
│  For quick calculations:                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1 day ≈ 10^5 seconds                                       │ │
│  │ 1 year ≈ 3 × 10^7 seconds                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Latency Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                   Latency Reference Table                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Operation                      │ Latency                       │
│  ───────────────────────────────────────────────────────────    │
│  L1 cache reference             │ 0.5 ns                        │
│  L2 cache reference             │ 7 ns                          │
│  Main memory reference          │ 100 ns                        │
│  SSD random read                │ 150 μs                        │
│  HDD disk seek                  │ 10 ms                         │
│  Same datacenter network RTT    │ 0.5 ms                        │
│  Different region network RTT   │ 150 ms                        │
│                                                                  │
│  Visualization (1 ns = 1 second):                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ L1 cache: 0.5 seconds                                      │ │
│  │ Main memory: 100 seconds (1 min 40 sec)                    │ │
│  │ SSD: 150,000 seconds (about 2 days)                        │ │
│  │ HDD: 10,000,000 seconds (about 4 months)                   │ │
│  │ Network (same DC): 500,000 seconds (about 6 days)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Availability Numbers (9's)

| Availability | Annual Downtime | Monthly Downtime |
|--------|--------------|--------------|
| 99% (two 9s) | 3.65 days | 7.3 hours |
| 99.9% (three 9s) | 8.77 hours | 43.8 minutes |
| 99.99% (four 9s) | 52.6 minutes | 4.38 minutes |
| 99.999% (five 9s) | 5.26 minutes | 26 seconds |

### 5.5 Typical Service Throughput

```
┌─────────────────────────────────────────────────────────────────┐
│               Service Throughput Reference                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Web Server (Nginx)             │ 10,000 ~ 100,000 req/s        │
│  Database (MySQL)               │ 10,000 ~ 50,000 QPS           │
│  Cache (Redis)                  │ 100,000 ~ 500,000 ops/s       │
│  Message Queue (Kafka)          │ 1,000,000 msg/s               │
│                                                                  │
│  Single server estimates:                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Web server: ~1,000 concurrent connections                │ │
│  │ • Database: ~10,000 QPS                                    │ │
│  │ • Cache: ~100,000 ops/s                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Practice Problems

### Problem 1: QPS Calculation

Calculate image upload QPS for an Instagram-like service.

Conditions:
- DAU: 500 million
- Daily image upload rate: 10% of users upload average 2 images

### Problem 2: Storage Estimation

Estimate 1-year message storage for a chat app.

Conditions:
- DAU: 100 million
- Average daily messages: 50/user
- Average message size: 100 bytes

### Problem 3: Server Count Estimation

Estimate web server count needed to handle 100,000 QPS.

Conditions:
- Single server throughput: 1,000 QPS
- Need 20% overhead for availability

### Problem 4: Requirements Clarification

Given "Design a URL shortening service", write 5 questions to ask the interviewer.

### Problem 5: System Design Practice

Draw high-level architecture for a simple file sharing service.

---

## Answers

### Problem 1 Answer

```
Images uploaded/day = 500M × 10% × 2 = 100M
Average QPS = 100M / 86,400 ≈ 1,160 QPS
Peak QPS = 1,160 × 3 ≈ 3,500 QPS
```

### Problem 2 Answer

```
Daily messages = 100M × 50 = 5B
Daily storage = 5B × 100B = 500GB
Annual storage = 500GB × 365 ≈ 180TB
```

### Problem 3 Answer

```
Base servers needed = 100,000 / 1,000 = 100
With overhead = 100 × 1.2 = 120
Consider HA (redundancy) = 120 × 2 = 240
```

### Problem 4 Answer

1. What are expected DAU and MAU?
2. Expected length/format of shortened URLs?
3. Do we need URL expiration feature?
4. Do we need custom short URL support?
5. Do we need analytics (click count stats)?

### Problem 5 Answer

```
┌─────────┐     ┌─────────────┐     ┌──────────────┐
│ Client  │────▶│ Load        │────▶│ Web Server   │
└─────────┘     │ Balancer    │     └──────┬───────┘
                └─────────────┘            │
                                          ┌┴─────────────┐
                                          │              │
                                          ▼              ▼
                                   ┌──────────┐  ┌──────────┐
                                   │ Metadata │  │ Object   │
                                   │ DB       │  │ Storage  │
                                   └──────────┘  └──────────┘
```

---

## 7. References

### Books
- System Design Interview - Alex Xu
- Designing Data-Intensive Applications - Martin Kleppmann

### Online Resources
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [ByteByteGo](https://bytebytego.com/)
- [High Scalability](http://highscalability.com/)

### Practice Sites
- [Pramp](https://www.pramp.com/) - Mock interviews
- [Interviewing.io](https://interviewing.io/)

---

**Next**: [Scalability Basics](./02_Scalability_Basics.md)

---

**Document Information**
- Last Updated: 2024
- Difficulty: ⭐
- Estimated Study Time: 2 hours
