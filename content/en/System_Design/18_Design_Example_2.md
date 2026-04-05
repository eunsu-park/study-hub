# Practical Design Examples 2

**Previous**: [Practical Design Examples 1](./17_Design_Example_1.md) | **Next**: [Observability and Monitoring](./19_Observability_Monitoring.md)

Difficulty: ⭐⭐⭐⭐

## Learning Objectives

1. Design a news feed/timeline system addressing fan-out strategies (push vs pull vs hybrid) and ranking algorithms
2. Design a real-time chat system handling WebSocket connections, message ordering, delivery receipts, and offline storage
3. Design a notification system supporting multiple channels (push, email, SMS), template management, and user preference handling
4. Analyze the trade-offs between precomputation (fan-out on write) and on-demand computation (fan-out on read) for social graph workloads
5. Apply connection management techniques (heartbeats, reconnection, presence detection) for persistent WebSocket architectures
6. Evaluate delivery guarantee requirements across different real-time system types and choose appropriate strategies

---

The systems in this lesson -- news feeds, chat, and notifications -- are among the most user-facing and latency-sensitive services at any social platform. Facebook's news feed serves over 2 billion users with personalized, ranked content; WhatsApp delivers 100 billion messages per day with end-to-end encryption; and every major app relies on a notification system to re-engage users across push, email, and SMS channels. Designing these systems teaches you to think about fan-out at scale, real-time delivery, and the delicate balance between freshness and compute cost.

## Table of Contents

1. [News Feed / Timeline](#1-news-feed--timeline)
2. [Chat System](#2-chat-system)
3. [Notification System](#3-notification-system)
4. [Practice Problems](#4-practice-problems)

---

## 1. News Feed / Timeline

### 1.1 Requirements Definition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Functional Requirements                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Core Features:                                                        │
│  1. Create posts: text, images, videos                                 │
│  2. News feed view: posts from followed users                          │
│  3. Timeline view: list of posts from a specific user                  │
│                                                                         │
│  Additional Features:                                                  │
│  4. Likes, comments                                                    │
│  5. Infinite scroll                                                    │
│  6. Real-time updates                                                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                     Scale                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  - Daily Active Users (DAU): 100M                                      │
│  - Average follows per user: 200                                       │
│  - Daily posts: 10M                                                    │
│  - Feed view frequency: 10 times/day/user                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Challenge: Push vs Pull

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Feed Generation Strategy                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Method 1: Pull (Fan-out on Read)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  When user requests feed:                                        │   │
│  │                                                                  │   │
│  │  1. Get following list                                          │   │
│  │  2. Get latest posts from each followed user                    │   │
│  │  3. Merge and sort                                              │   │
│  │  4. Return                                                       │   │
│  │                                                                  │   │
│  │  ┌──────────┐                                                   │   │
│  │  │   User   │─── GET /feed                                      │   │
│  │  └────┬─────┘                                                   │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  SELECT * FROM posts                                       │ │   │
│  │  │  WHERE author_id IN (SELECT followee FROM follows          │ │   │
│  │  │                      WHERE follower = user_id)             │ │   │
│  │  │  ORDER BY created_at DESC                                  │ │   │
│  │  │  LIMIT 20;                                                 │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  Pros:                                                          │   │
│  │  - Saves storage space                                          │   │
│  │  - Post creation is fast                                        │   │
│  │                                                                  │   │
│  │  Cons:                                                          │   │
│  │  - Feed retrieval is slow (if following many)                   │   │
│  │  - DB load                                                       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Method 2: Push (Fan-out on Write)                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  When post is created:                                          │   │
│  │                                                                  │   │
│  │  1. Save post                                                   │   │
│  │  2. Add to all followers' feeds                                 │   │
│  │                                                                  │   │
│  │  ┌──────────┐                                                   │   │
│  │  │  Author  │─── POST /posts                                    │   │
│  │  └────┬─────┘                                                   │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  ┌──────────────┐                                               │   │
│  │  │ Save Post    │                                               │   │
│  │  └──────┬───────┘                                               │   │
│  │         │                                                        │   │
│  │         ▼                                                        │   │
│  │  ┌──────────────┐     ┌──────────────────────────────────────┐  │   │
│  │  │ Fan-out to   │────►│ Follower 1 Feed: [post_id, ...]      │  │   │
│  │  │ all followers│────►│ Follower 2 Feed: [post_id, ...]      │  │   │
│  │  │              │────►│ Follower 3 Feed: [post_id, ...]      │  │   │
│  │  │              │────►│ ...                                   │  │   │
│  │  └──────────────┘     └──────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  Pros:                                                          │   │
│  │  - Feed retrieval is fast (pre-computed)                        │   │
│  │                                                                  │   │
│  │  Cons:                                                          │   │
│  │  - Uses more storage space                                      │   │
│  │  - Celeb (high follower count) post delays                      │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Hybrid Fan-out                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Key Idea:                                                             │
│  - Regular users: Push (Fan-out on Write)                              │
│  - Hot users (celebrities): Pull (Fan-out on Read)                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Post Creation:                                                  │   │
│  │                                                                  │   │
│  │  ┌──────────┐                                                   │   │
│  │  │  Author  │──► Check follower count                           │   │
│  │  └────┬─────┘                                                   │   │
│  │       │                                                          │   │
│  │  ┌────┴────┐                                                     │   │
│  │  │         │                                                     │   │
│  │  ▼         ▼                                                     │   │
│  │ < 10K    ≥ 10K                                                  │   │
│  │ followers followers                                              │   │
│  │  │         │                                                     │   │
│  │  ▼         ▼                                                     │   │
│  │ Push     Save to                                                │   │
│  │ to all   hot_posts                                              │   │
│  │ feeds    table only                                             │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Feed Retrieval:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. Get pre-computed feed (pushed posts)                        │   │
│  │  2. Get latest posts from followed hot users                    │   │
│  │  3. Merge and sort                                              │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                                                          │   │   │
│  │  │  Pre-computed Feed     Hot Users Posts                   │   │   │
│  │  │  ┌───────────────┐    ┌───────────────┐                 │   │   │
│  │  │  │ [post1]       │    │ [celeb_post1] │                 │   │   │
│  │  │  │ [post2]       │ +  │ [celeb_post2] │                 │   │   │
│  │  │  │ [post3]       │    │               │                 │   │   │
│  │  │  └───────────────┘    └───────────────┘                 │   │   │
│  │  │           │                    │                         │   │   │
│  │  │           └────────┬───────────┘                         │   │   │
│  │  │                    ▼                                     │   │   │
│  │  │             Merge & Sort                                 │   │   │
│  │  │                    │                                     │   │   │
│  │  │                    ▼                                     │   │   │
│  │  │            Final Feed                                    │   │   │
│  │  │                                                          │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     News Feed Architecture                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │   ┌────────┐                                                     │ │
│  │   │ Client │                                                     │ │
│  │   └───┬────┘                                                     │ │
│  │       │                                                          │ │
│  │       ▼                                                          │ │
│  │   ┌─────────────────┐                                            │ │
│  │   │   API Gateway   │                                            │ │
│  │   └───────┬─────────┘                                            │ │
│  │           │                                                      │ │
│  │   ┌───────┴───────┐                                              │ │
│  │   │               │                                              │ │
│  │   ▼               ▼                                              │ │
│  │ ┌─────────────┐ ┌─────────────┐                                 │ │
│  │ │ Post Service│ │ Feed Service│                                 │ │
│  │ └──────┬──────┘ └──────┬──────┘                                 │ │
│  │        │               │                                        │ │
│  │        ▼               │                                        │ │
│  │ ┌─────────────┐        │                                        │ │
│  │ │   Posts DB  │        │                                        │ │
│  │ │  (Sharded)  │        │                                        │ │
│  │ └──────┬──────┘        │                                        │ │
│  │        │               │                                        │ │
│  │        ▼               ▼                                        │ │
│  │ ┌──────────────────────────────────────┐                        │ │
│  │ │           Message Queue              │                        │ │
│  │ │            (Kafka)                   │                        │ │
│  │ └─────────────────┬────────────────────┘                        │ │
│  │                   │                                              │ │
│  │                   ▼                                              │ │
│  │ ┌──────────────────────────────────────┐                        │ │
│  │ │        Fanout Workers                │                        │ │
│  │ │  ┌────────┐ ┌────────┐ ┌────────┐   │                        │ │
│  │ │  │Worker 1│ │Worker 2│ │Worker 3│   │                        │ │
│  │ │  └───┬────┘ └───┬────┘ └───┬────┘   │                        │ │
│  │ └──────┼──────────┼──────────┼────────┘                        │ │
│  │        │          │          │                                  │ │
│  │        └──────────┼──────────┘                                  │ │
│  │                   ▼                                              │ │
│  │ ┌──────────────────────────────────────┐                        │ │
│  │ │         Feed Cache (Redis)           │                        │ │
│  │ │   user:123:feed = [post_ids...]      │                        │ │
│  │ └──────────────────────────────────────┘                        │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.5 Feed Caching Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Feed Caching                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Redis Feed Structure:                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Key: feed:{user_id}                                            │   │
│  │  Type: Sorted Set                                               │   │
│  │  Score: timestamp                                               │   │
│  │  Member: post_id                                                │   │
│  │                                                                  │   │
│  │  ZADD feed:123 1704067200 "post_abc"                           │   │
│  │  ZADD feed:123 1704067300 "post_def"                           │   │
│  │  ...                                                            │   │
│  │                                                                  │   │
│  │  Query: ZREVRANGE feed:123 0 19                                 │   │
│  │  → Returns 20 most recent post IDs                              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Cache Management:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  - Limit feed size: Keep only last 800 posts                   │   │
│  │    ZREMRANGEBYRANK feed:123 0 -801                              │   │
│  │                                                                  │   │
│  │  - TTL setting: Cache only active users                         │   │
│  │    Inactive users → Rebuild on query                            │   │
│  │                                                                  │   │
│  │  - Post content in separate cache                               │   │
│  │    Key: post:{post_id}                                          │   │
│  │    Value: { author, content, media, ... }                       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chat System

### 2.1 Requirements Definition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Functional Requirements                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Core Features:                                                        │
│  1. 1:1 chat                                                           │
│  2. Group chat (up to 500 members)                                     │
│  3. Online status indicator                                            │
│  4. Read receipts                                                      │
│                                                                         │
│  Message Features:                                                     │
│  5. Text, image, file transfer                                         │
│  6. Message history sync                                               │
│  7. Push notifications (when offline)                                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                     Non-Functional Requirements                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  - Real-time delivery: < 100ms                                         │
│  - Message ordering guarantee                                          │
│  - No message loss                                                     │
│  - Concurrent connections: millions                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Communication Protocol

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Protocol Selection                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. HTTP Polling                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Client              Server                                     │   │
│  │    │                   │                                        │   │
│  │    │── GET /messages ─►│                                        │   │
│  │    │◄── [] ────────────│                                        │   │
│  │    │                   │                                        │   │
│  │    │── GET /messages ─►│ (5 seconds later)                     │   │
│  │    │◄── [] ────────────│                                        │   │
│  │    │                   │                                        │   │
│  │    │── GET /messages ─►│ (5 seconds later)                     │   │
│  │    │◄── [msg1] ────────│                                        │   │
│  │                                                                  │   │
│  │  Cons: Latency, unnecessary requests, server load              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. Long Polling                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Client              Server                                     │   │
│  │    │                   │                                        │   │
│  │    │── GET /messages ─►│                                        │   │
│  │    │    (waiting...)   │                                        │   │
│  │    │                   │ Message arrives!                       │   │
│  │    │◄── [msg1] ────────│                                        │   │
│  │    │── GET /messages ─►│ (immediate reconnect)                 │   │
│  │    │    (waiting...)   │                                        │   │
│  │                                                                  │   │
│  │  Improvement: Reduced unnecessary requests                      │   │
│  │  Cons: Connection overhead, server resources                    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  3. WebSocket (Recommended)                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Client              Server                                     │   │
│  │    │                   │                                        │   │
│  │    │── HTTP Upgrade ──►│                                        │   │
│  │    │◄─ 101 Switching ──│                                        │   │
│  │    │                   │                                        │   │
│  │    │═══ WebSocket ═════│ (bidirectional connection maintained) │   │
│  │    │                   │                                        │   │
│  │    │◄── [msg1] ────────│ (server push)                         │   │
│  │    │── [msg2] ────────►│ (client send)                         │   │
│  │    │◄── [msg3] ────────│                                        │   │
│  │                                                                  │   │
│  │  Pros: Real-time, bidirectional, minimal overhead              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Chat System Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                    │ │
│  │  │ Client A │    │ Client B │    │ Client C │                    │ │
│  │  └────┬─────┘    └────┬─────┘    └────┬─────┘                    │ │
│  │       │ WebSocket     │ WebSocket     │ WebSocket                │ │
│  │       │               │               │                          │ │
│  │       ▼               ▼               ▼                          │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                   Load Balancer                             │ │ │
│  │  │              (Sticky Sessions by user_id)                   │ │ │
│  │  └──────────────────────┬──────────────────────────────────────┘ │ │
│  │                         │                                        │ │
│  │       ┌─────────────────┼─────────────────┐                     │ │
│  │       ▼                 ▼                 ▼                     │ │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐               │ │
│  │  │ Chat Srv 1 │   │ Chat Srv 2 │   │ Chat Srv 3 │               │ │
│  │  │ [A's conn] │   │ [B's conn] │   │ [C's conn] │               │ │
│  │  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘               │ │
│  │        │                │                │                       │ │
│  │        └────────────────┼────────────────┘                       │ │
│  │                         │                                        │ │
│  │                         ▼                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │              Message Broker (Redis Pub/Sub)                 │ │ │
│  │  │                                                             │ │ │
│  │  │  A sends to B:                                              │ │ │
│  │  │  1. Chat Srv 1 → PUBLISH chat:B "msg from A"               │ │ │
│  │  │  2. Chat Srv 2 ← SUBSCRIBE chat:B → deliver to B           │ │ │
│  │  │                                                             │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                         │                                        │ │
│  │                         ▼                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │               Message Store (Cassandra)                     │ │ │
│  │  │                                                             │ │ │
│  │  │  Partition Key: conversation_id                            │ │ │
│  │  │  Clustering Key: message_id (time-based)                   │ │ │
│  │  │                                                             │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Message Delivery Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     1:1 Message Delivery                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User A → User B message delivery:                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. A ─── [msg] ───► Chat Server 1                              │   │
│  │                          │                                       │   │
│  │  2.                      ▼                                       │   │
│  │           ┌─────────────────────────────┐                       │   │
│  │           │ Message Service             │                       │   │
│  │           │ - Generate message_id       │                       │   │
│  │           │ - Validate message          │                       │   │
│  │           └─────────────┬───────────────┘                       │   │
│  │                         │                                        │   │
│  │  3.                     │                                        │   │
│  │           ┌─────────────┼───────────────┐                       │   │
│  │           │             │               │                       │   │
│  │           ▼             ▼               ▼                       │   │
│  │     ┌──────────┐  ┌──────────┐   ┌──────────────┐              │   │
│  │     │ Store    │  │ Publish  │   │ Check B      │              │   │
│  │     │ to DB    │  │ to Kafka │   │ online?      │              │   │
│  │     └──────────┘  └──────────┘   └──────┬───────┘              │   │
│  │                                         │                       │   │
│  │  4.                              ┌──────┴──────┐                │   │
│  │                                  │             │                │   │
│  │                               Online        Offline             │   │
│  │                                  │             │                │   │
│  │                                  ▼             ▼                │   │
│  │                           ┌──────────┐  ┌──────────┐           │   │
│  │                           │ Pub/Sub  │  │  Push    │           │   │
│  │                           │ to B     │  │  Queue   │           │   │
│  │                           └────┬─────┘  └──────────┘           │   │
│  │                                │                                │   │
│  │  5.                            ▼                                │   │
│  │                     Chat Server 2 ─── [msg] ───► B              │   │
│  │                                                                  │   │
│  │  6. A ◄─── [ack] ─── Chat Server 1                              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Group Chat

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Group Chat Design                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Group Message Delivery:                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. Sender ─── [msg] ───► Message Service                       │   │
│  │                               │                                  │   │
│  │  2.                           ▼                                  │   │
│  │                    Get group members (100 members)              │   │
│  │                               │                                  │   │
│  │  3.                           ▼                                  │   │
│  │              ┌────────────────┼────────────────┐                │   │
│  │              │                │                │                │   │
│  │              ▼                ▼                ▼                │   │
│  │         [Online 60]     [Online 30]     [Offline 10]           │   │
│  │              │                │                │                │   │
│  │              ▼                ▼                ▼                │   │
│  │         Pub/Sub          Pub/Sub         Push Queue            │   │
│  │         (batch)          (batch)                                │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Optimizations:                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. Prioritize online members                                   │   │
│  │  2. Batch processing (multiple members at once)                 │   │
│  │  3. Read receipts with sampling (not all members)               │   │
│  │  4. Large groups: Consider client-side pull approach            │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.6 Online Status Management

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Online Presence                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Status Types:                                                         │
│  - Online: Currently connected                                         │
│  - Offline: No connection                                              │
│  - Away: Connected but inactive                                        │
│  - Last Seen: Last connection time                                     │
│                                                                         │
│  Implementation:                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. Heartbeat-based                                              │   │
│  │                                                                  │   │
│  │  Client ─── heartbeat (every 30 sec) ───► Server                │   │
│  │                                              │                   │   │
│  │                                              ▼                   │   │
│  │                              Redis: SET presence:user123 "online"│   │
│  │                                     EXPIRE presence:user123 60   │   │
│  │                                                                  │   │
│  │  No heartbeat for 60 sec → Key expires → Offline                │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Status Propagation:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  When User A's status changes:                                  │   │
│  │                                                                  │   │
│  │  1. Get A's friends list                                        │   │
│  │  2. Filter only online friends                                  │   │
│  │  3. Push status change to those friends                         │   │
│  │                                                                  │   │
│  │  Optimizations:                                                  │   │
│  │  - Batch processing if many friends                             │   │
│  │  - Prevent frequent changes (debounce)                          │   │
│  │  - Prioritize active chat participants                          │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.7 Message Storage

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Message DB Schema (Cassandra)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  messages table:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  CREATE TABLE messages (                                        │   │
│  │      conversation_id UUID,         -- Conversation ID          │   │
│  │      message_id TIMEUUID,          -- Time-based UUID          │   │
│  │      sender_id UUID,               -- Sender                    │   │
│  │      content TEXT,                 -- Message content          │   │
│  │      content_type TEXT,            -- text, image, file         │   │
│  │      created_at TIMESTAMP,         -- Creation time            │   │
│  │      PRIMARY KEY (conversation_id, message_id)                  │   │
│  │  ) WITH CLUSTERING ORDER BY (message_id DESC);                  │   │
│  │                                                                  │   │
│  │  -- Sorted by latest message first                              │   │
│  │  -- Partitioned by conversation (watch for hot partitions)      │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  conversations table:                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  CREATE TABLE user_conversations (                              │   │
│  │      user_id UUID,                                              │   │
│  │      last_message_at TIMESTAMP,    -- For sorting              │   │
│  │      conversation_id UUID,                                      │   │
│  │      conversation_type TEXT,       -- dm, group                 │   │
│  │      unread_count INT,             -- Unread messages          │   │
│  │      PRIMARY KEY (user_id, last_message_at, conversation_id)    │   │
│  │  ) WITH CLUSTERING ORDER BY (last_message_at DESC);             │   │
│  │                                                                  │   │
│  │  -- Query conversations per user                                │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Notification System

### 3.1 Requirements Definition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Functional Requirements                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Notification Channels:                                                │
│  1. iOS push (APNs)                                                    │
│  2. Android push (FCM)                                                 │
│  3. SMS                                                                │
│  4. Email                                                              │
│                                                                         │
│  Features:                                                             │
│  5. Notification templates                                             │
│  6. Per-user notification settings                                     │
│  7. Scheduled notifications                                            │
│  8. Notification history                                               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                     Non-Functional Requirements                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  - Daily notifications: 10B (10 billion)                               │
│  - Soft real-time: Delivery within seconds                            │
│  - Deduplication                                                       │
│  - Priority support                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Notification System Architecture                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Event Sources                            │ │ │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │ │ │
│  │  │  │Order   │ │Payment │ │Social  │ │Schedule│              │ │ │
│  │  │  │Service │ │Service │ │Service │ │Service │              │ │ │
│  │  │  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘              │ │ │
│  │  │      │          │          │          │                    │ │ │
│  │  └──────┼──────────┼──────────┼──────────┼────────────────────┘ │ │
│  │         │          │          │          │                      │ │
│  │         └──────────┴──────────┴──────────┘                      │ │
│  │                         │                                        │ │
│  │                         ▼                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                 Notification Service                        │ │ │
│  │  │  ┌──────────────────────────────────────────────────────┐  │ │ │
│  │  │  │ 1. Validation                                         │  │ │ │
│  │  │  │ 2. User Preferences Check                             │  │ │ │
│  │  │  │ 3. Rate Limiting                                      │  │ │ │
│  │  │  │ 4. Template Rendering                                 │  │ │ │
│  │  │  │ 5. Priority Assignment                                │  │ │ │
│  │  │  └──────────────────────────────────────────────────────┘  │ │ │
│  │  └──────────────────────┬──────────────────────────────────────┘ │ │
│  │                         │                                        │ │
│  │                         ▼                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │              Message Queues (Priority-based)                │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │ │ │
│  │  │  │ High Queue  │ │ Medium Queue│ │ Low Queue   │           │ │ │
│  │  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │ │ │
│  │  └─────────┼───────────────┼───────────────┼───────────────────┘ │ │
│  │            │               │               │                     │ │
│  │            ▼               ▼               ▼                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                     Workers                                 │ │ │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │ │ │
│  │  │  │  iOS   │ │Android │ │  SMS   │ │ Email  │              │ │ │
│  │  │  │ Worker │ │ Worker │ │ Worker │ │ Worker │              │ │ │
│  │  │  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘              │ │ │
│  │  └──────┼──────────┼──────────┼──────────┼─────────────────────┘ │ │
│  │         │          │          │          │                       │ │
│  │         ▼          ▼          ▼          ▼                       │ │
│  │      ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                     │ │
│  │      │ APNs │  │ FCM  │  │Twilio│  │ SES  │                     │ │
│  │      └──────┘  └──────┘  └──────┘  └──────┘                     │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Notification Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Notification Delivery Flow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. Receive Event                                               │   │
│  │  ─────────────                                                   │   │
│  │  {                                                               │   │
│  │    "event_type": "order_shipped",                               │   │
│  │    "user_id": "user123",                                        │   │
│  │    "data": { "order_id": "ORD456", "tracking": "..." }          │   │
│  │  }                                                               │   │
│  │                                                                  │   │
│  │  2. Check User Settings                                         │   │
│  │  ─────────────────                                               │   │
│  │  SELECT * FROM user_notification_settings                       │   │
│  │  WHERE user_id = 'user123';                                     │   │
│  │                                                                  │   │
│  │  → push: true, email: true, sms: false                          │   │
│  │                                                                  │   │
│  │  3. Get Device Tokens                                           │   │
│  │  ─────────────────                                               │   │
│  │  SELECT device_token, platform                                  │   │
│  │  FROM user_devices WHERE user_id = 'user123';                   │   │
│  │                                                                  │   │
│  │  → [{ token: "abc...", platform: "ios" },                       │   │
│  │      { token: "def...", platform: "android" }]                  │   │
│  │                                                                  │   │
│  │  4. Render Template                                             │   │
│  │  ─────────────────                                               │   │
│  │  Template: "Your order {order_id} has been shipped!"            │   │
│  │  Result: "Your order ORD456 has been shipped!"                  │   │
│  │                                                                  │   │
│  │  5. Deduplication Check                                         │   │
│  │  ─────────────────                                               │   │
│  │  Redis SETNX dedup:{event_hash} 1 EX 86400                      │   │
│  │  → Prevent duplicate notifications within 24 hours              │   │
│  │                                                                  │   │
│  │  6. Publish to Queues                                           │   │
│  │  ─────────────────                                               │   │
│  │  Publish to ios_queue, android_queue, email_queue               │   │
│  │                                                                  │   │
│  │  7. Worker Processing                                           │   │
│  │  ─────────────────                                               │   │
│  │  iOS Worker → APNs API → Apple servers                          │   │
│  │  Android Worker → FCM API → Google servers                      │   │
│  │  Email Worker → SES API → Email delivery                        │   │
│  │                                                                  │   │
│  │  8. Store Result                                                │   │
│  │  ─────────────────                                               │   │
│  │  INSERT INTO notification_logs (...)                            │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Deduplication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Duplicate Notification Prevention                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problems:                                                             │
│  - Duplicate event publishing                                          │
│  - Duplicates from worker retries                                      │
│  - Race conditions in distributed environment                          │
│                                                                         │
│  Solutions:                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. Event-Level Deduplication                                   │   │
│  │  ─────────────────────────                                       │   │
│  │  event_key = hash(event_type + user_id + key_data)              │   │
│  │                                                                  │   │
│  │  if not redis.setnx(f"dedup:{event_key}", 1, ex=86400):        │   │
│  │      return  # Already processed                                │   │
│  │                                                                  │   │
│  │  2. Delivery-Level Deduplication                                │   │
│  │  ─────────────────────────                                       │   │
│  │  notification_id = generate_unique_id()                         │   │
│  │                                                                  │   │
│  │  Check before each channel delivery:                            │   │
│  │  if not redis.setnx(f"sent:{notification_id}:{channel}", 1):   │   │
│  │      return  # Already sent                                     │   │
│  │                                                                  │   │
│  │  3. Frequency Limiting                                          │   │
│  │  ─────────────────────────                                       │   │
│  │  max_per_hour = 10                                              │   │
│  │                                                                  │   │
│  │  current = redis.incr(f"rate:{user_id}:{hour}")                │   │
│  │  if current > max_per_hour:                                     │   │
│  │      # Limit exceeded → send later or ignore                    │   │
│  │      enqueue_for_later()                                        │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Priority Handling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Notification Priority                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Priority Classification:                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  HIGH (immediate):                                               │   │
│  │  - Security alerts (suspicious login)                           │   │
│  │  - 2FA codes                                                    │   │
│  │  - Urgent system alerts                                         │   │
│  │                                                                  │   │
│  │  MEDIUM (within seconds):                                       │   │
│  │  - Chat messages                                                │   │
│  │  - Order status changes                                         │   │
│  │  - Payment notifications                                        │   │
│  │                                                                  │   │
│  │  LOW (minutes to hours):                                        │   │
│  │  - Marketing notifications                                      │   │
│  │  - Recommendations                                              │   │
│  │  - Summary notifications                                        │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Implementation:                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Priority-based queues:                                         │   │
│  │                                                                  │   │
│  │  ┌─────────────────┐                                            │   │
│  │  │ HIGH Queue      │ ──► Dedicated 10 Workers                   │   │
│  │  └─────────────────┘     (always process immediately)           │   │
│  │                                                                  │   │
│  │  ┌─────────────────┐                                            │   │
│  │  │ MEDIUM Queue    │ ──► Shared 50 Workers                      │   │
│  │  └─────────────────┘     (when HIGH is empty)                   │   │
│  │                                                                  │   │
│  │  ┌─────────────────┐                                            │   │
│  │  │ LOW Queue       │ ──► Shared 50 Workers                      │   │
│  │  └─────────────────┘     (when HIGH, MEDIUM empty)              │   │
│  │                                                                  │   │
│  │  Worker processing order:                                       │   │
│  │  while True:                                                    │   │
│  │      msg = high_queue.pop() or                                  │   │
│  │            medium_queue.pop() or                                │   │
│  │            low_queue.pop()                                      │   │
│  │      process(msg)                                               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.6 Notification Settings Schema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DB Schema                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  user_notification_settings:                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  user_id          UUID          PK                              │   │
│  │  channel          VARCHAR       PK (push/email/sms)             │   │
│  │  enabled          BOOLEAN                                       │   │
│  │  quiet_hours      JSONB         {"start": "22:00", "end": "08:00"}│  │
│  │  frequency        VARCHAR       immediate/daily/weekly          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  notification_type_settings:                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  user_id          UUID          PK                              │   │
│  │  notification_type VARCHAR      PK (order/social/marketing)     │   │
│  │  push_enabled     BOOLEAN                                       │   │
│  │  email_enabled    BOOLEAN                                       │   │
│  │  sms_enabled      BOOLEAN                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  user_devices:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  device_id        UUID          PK                              │   │
│  │  user_id          UUID          FK                              │   │
│  │  platform         VARCHAR       ios/android                     │   │
│  │  device_token     VARCHAR       Push token                      │   │
│  │  last_active      TIMESTAMP     Last activity                   │   │
│  │  app_version      VARCHAR       App version                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  notification_logs:                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  id               UUID          PK                              │   │
│  │  user_id          UUID                                          │   │
│  │  type             VARCHAR       Notification type               │   │
│  │  channel          VARCHAR       Delivery channel                │   │
│  │  content          JSONB         Notification content            │   │
│  │  status           VARCHAR       sent/failed/pending             │   │
│  │  sent_at          TIMESTAMP     Sent time                       │   │
│  │  read_at          TIMESTAMP     Read time                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Practice Problems

### Exercise 1: News Feed Extension

Design additions with the following features:
- Ad insertion (every 5th post)
- Trending post recommendations
- "Not interested" feedback handling

### Exercise 2: Chat System Extension

Design to meet the following requirements:
- End-to-End encryption
- Message edit/delete (within 24 hours)
- Video/voice call signaling

### Exercise 3: Notification System Optimization

Design to handle the following scenarios:
- Global service: Multi-language notifications
- Notification batching: Grouping similar notifications
- A/B testing: Notification copy optimization

---

**Previous**: [Practical Design Examples 1](./17_Design_Example_1.md) | **Next**: [Observability and Monitoring](./19_Observability_Monitoring.md)

---

## References

- "System Design Interview" - Alex Xu Vol.1 & Vol.2
- "Designing Data-Intensive Applications" - Martin Kleppmann
- Twitter Timeline Architecture
- Facebook News Feed Architecture
- WhatsApp Architecture at Scale
- Discord How Discord Stores Billions of Messages
- Airbnb Notification System
