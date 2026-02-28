[Previous: Software-Defined Networking](./20_Software_Defined_Networking.md) | [Next: Multicast](./22_Multicast.md)

---

# 21. Quality of Service (QoS)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why QoS is needed and distinguish between IntServ and DiffServ architectures
2. Classify network traffic and map it to appropriate QoS classes
3. Implement traffic policing (token bucket) and shaping algorithms
4. Describe queuing disciplines (FIFO, WFQ, CBWFQ, LLQ) and their trade-offs
5. Apply DSCP markings and configure QoS policies for voice, video, and data

---

## Table of Contents

1. [Why QoS?](#1-why-qos)
2. [Traffic Classification](#2-traffic-classification)
3. [IntServ vs DiffServ](#3-intserv-vs-diffserv)
4. [Traffic Policing and Shaping](#4-traffic-policing-and-shaping)
5. [Queuing Disciplines](#5-queuing-disciplines)
6. [DSCP and Per-Hop Behavior](#6-dscp-and-per-hop-behavior)
7. [QoS Design Patterns](#7-qos-design-patterns)
8. [Exercises](#8-exercises)

---

## 1. Why QoS?

### 1.1 Best-Effort Limitations

IP networks are best-effort by default — no guarantees on bandwidth, latency, jitter, or packet loss. This works for email and web browsing but fails for:

```
Application Requirements:

  Voice (VoIP):
    Latency  < 150ms (one-way)
    Jitter   < 30ms
    Loss     < 1%
    Bandwidth: ~100 Kbps per call

  Video conferencing:
    Latency  < 200ms
    Jitter   < 50ms
    Loss     < 0.1% (I-frames critical)
    Bandwidth: 1-5 Mbps

  File transfer:
    Latency  doesn't matter
    Jitter   doesn't matter
    Loss     0% (TCP retransmits)
    Bandwidth: as much as possible

Without QoS: a large file transfer can starve voice calls.
```

### 1.2 QoS Mechanisms Overview

```
End-to-end QoS pipeline:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Classify │───►│  Police  │───►│  Queue   │───►│ Schedule │
  │ & Mark   │    │ & Shape  │    │          │    │ & Send   │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘

  1. Classify: Identify traffic type (voice, video, data)
  2. Mark: Set DSCP/ToS bits in IP header
  3. Police/Shape: Limit traffic to agreed rates
  4. Queue: Separate queues per traffic class
  5. Schedule: Service queues with priority/weight
```

---

## 2. Traffic Classification

### 2.1 Classification Methods

| Method | How | Pros | Cons |
|--------|-----|------|------|
| Port-based | Match TCP/UDP port (80=HTTP, 5060=SIP) | Simple, fast | Unreliable (ports can change) |
| Protocol-based | Match IP protocol field | Standard | Coarse-grained |
| DSCP-based | Read DSCP marking in IP header | Efficient | Requires trust at ingress |
| NBAR/DPI | Deep packet inspection | Accurate | CPU-intensive, privacy concerns |
| Flow-based | 5-tuple (src/dst IP, src/dst port, protocol) | Precise | Large state tables |

### 2.2 Common Traffic Classes

```python
# Typical enterprise QoS classification

QOS_CLASSES = {
    'voice': {
        'dscp': 46,          # EF (Expedited Forwarding)
        'bandwidth': '10%',
        'priority': True,     # Strict priority (LLQ)
        'description': 'VoIP RTP streams',
    },
    'video': {
        'dscp': 34,          # AF41
        'bandwidth': '30%',
        'priority': False,
        'description': 'Video conferencing',
    },
    'critical_data': {
        'dscp': 26,          # AF31
        'bandwidth': '25%',
        'priority': False,
        'description': 'Business applications (ERP, DB)',
    },
    'best_effort': {
        'dscp': 0,           # BE (Default)
        'bandwidth': '25%',
        'priority': False,
        'description': 'Web, email, general traffic',
    },
    'scavenger': {
        'dscp': 8,           # CS1
        'bandwidth': '10%',
        'priority': False,
        'description': 'Backup, updates, non-critical',
    },
}
```

---

## 3. IntServ vs DiffServ

### 3.1 Integrated Services (IntServ)

IntServ provides per-flow guarantees using RSVP (Resource Reservation Protocol):

```
IntServ with RSVP:

  Sender ─────────────────────────────────► Receiver
          ←─── RSVP PATH message ────────
          ────── RSVP RESV message ──────►

  Each router along the path:
    1. Receives reservation request
    2. Checks if resources available
    3. Reserves bandwidth/buffer for this flow
    4. Maintains per-flow state

  Guarantee: Bandwidth and delay bounds for this specific flow.
```

**Problem**: IntServ doesn't scale — routers must maintain state for every flow.

### 3.2 Differentiated Services (DiffServ)

DiffServ provides per-class (not per-flow) treatment using DSCP markings:

```
DiffServ:

  Edge routers:                  Core routers:
  ┌──────────────────────┐      ┌──────────────────────┐
  │ Classify traffic     │      │ Read DSCP marking    │
  │ Mark DSCP in header  │─────►│ Apply per-hop behavior│
  │ Police at ingress    │      │ (no per-flow state)  │
  └──────────────────────┘      └──────────────────────┘

  Scalable: core routers only look at 6-bit DSCP field.
  No per-flow state needed.
```

### 3.3 Comparison

| Feature | IntServ | DiffServ |
|---------|---------|----------|
| Granularity | Per-flow | Per-class |
| Guarantees | Hard (bandwidth, delay) | Soft (relative priority) |
| Scalability | Poor (per-flow state in core) | Excellent (stateless core) |
| Signaling | RSVP (complex) | DSCP marking (simple) |
| Deployment | Rare (too complex) | Standard (widely deployed) |

---

## 4. Traffic Policing and Shaping

### 4.1 Token Bucket Algorithm

Both policing and shaping use the token bucket:

```
Token Bucket:

  Tokens arrive at rate r (tokens/sec)
  Bucket holds max b tokens (burst size)

  ┌──────────────┐
  │ Token Bucket │ ← tokens arrive at rate r
  │  ┌────────┐  │
  │  │████████│  │ max capacity = b tokens
  │  │████████│  │
  │  │████    │  │ current tokens
  │  └────────┘  │
  └──────┬───────┘
         │
    ┌────┴────┐
    │ Packet  │  Each packet consumes tokens equal to its size.
    │ arrives │  If enough tokens → conform (send/mark green)
    └─────────┘  If not enough → exceed (drop/mark red)
```

### 4.2 Implementation

```python
import time


class TokenBucket:
    """Token bucket for traffic policing/shaping.

    Committed Information Rate (CIR): sustained rate
    Committed Burst Size (CBS): maximum burst in bytes
    """

    def __init__(self, cir_bps, cbs_bytes):
        self.cir = cir_bps          # bytes per second
        self.cbs = cbs_bytes        # bucket capacity
        self.tokens = cbs_bytes     # start full
        self.last_time = time.time()

    def consume(self, packet_size):
        """Try to consume tokens for a packet.

        Returns 'conform' if tokens available, 'exceed' otherwise.
        """
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now

        # Add tokens based on elapsed time
        self.tokens = min(
            self.cbs,
            self.tokens + elapsed * self.cir
        )

        if self.tokens >= packet_size:
            self.tokens -= packet_size
            return 'conform'
        else:
            return 'exceed'


class TrafficShaper:
    """Traffic shaper: delays excess packets instead of dropping.

    Unlike policing (which drops), shaping buffers packets
    and sends them when tokens become available.
    """

    def __init__(self, cir_bps, cbs_bytes, buffer_size=100):
        self.bucket = TokenBucket(cir_bps, cbs_bytes)
        self.buffer = []
        self.buffer_size = buffer_size

    def enqueue(self, packet):
        result = self.bucket.consume(packet['size'])
        if result == 'conform':
            return 'send', packet
        else:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(packet)
                return 'buffered', None
            else:
                return 'dropped', None  # buffer full

    def dequeue(self):
        """Send buffered packets when tokens available."""
        sent = []
        remaining = []
        for pkt in self.buffer:
            if self.bucket.consume(pkt['size']) == 'conform':
                sent.append(pkt)
            else:
                remaining.append(pkt)
        self.buffer = remaining
        return sent
```

### 4.3 Policing vs Shaping

| Aspect | Policing | Shaping |
|--------|----------|---------|
| Over-limit action | Drop or re-mark | Buffer and delay |
| Latency impact | None (no buffering) | Adds delay |
| Buffer needed | No | Yes |
| Packet loss | Higher | Lower (unless buffer overflows) |
| Use case | Ingress enforcement | Egress smoothing |

---

## 5. Queuing Disciplines

### 5.1 FIFO (First In, First Out)

Default: all packets in a single queue, served in arrival order. No differentiation.

### 5.2 Priority Queuing (PQ)

```
Priority Queuing:

  High priority:  ████████ → served first (always)
  Medium priority: ██████████████
  Low priority:    ████████████████████████

  Problem: starvation — low priority may never get served
  if high priority traffic is continuous.
```

### 5.3 Weighted Fair Queuing (WFQ)

```python
class WeightedFairQueue:
    """Weighted Fair Queuing: proportional bandwidth allocation.

    Each queue gets bandwidth proportional to its weight.
    Prevents starvation while providing differentiation.
    """

    def __init__(self, weights):
        """weights: dict of queue_name → weight (higher = more bandwidth)"""
        self.queues = {name: [] for name in weights}
        self.weights = weights
        self.total_weight = sum(weights.values())

    def enqueue(self, queue_name, packet):
        self.queues[queue_name].append(packet)

    def schedule(self, num_slots):
        """Decide which queues to serve in the next scheduling round.

        Returns list of (queue_name, packet) to send.
        """
        result = []
        for name, weight in self.weights.items():
            # Each queue gets slots proportional to its weight
            slots = max(1, round(num_slots * weight / self.total_weight))
            for _ in range(slots):
                if self.queues[name]:
                    result.append((name, self.queues[name].pop(0)))
        return result
```

### 5.4 Class-Based WFQ (CBWFQ) with Low-Latency Queuing (LLQ)

```
LLQ/CBWFQ (most common enterprise QoS):

  ┌────────────────────────────────────────────┐
  │               Scheduler                     │
  │                                            │
  │  ┌──────────────────┐  Strict Priority     │
  │  │ Voice (LLQ)      │──────────────────►   │
  │  │ DSCP EF          │  Always served first │
  │  │ policed to 10%   │                      │
  │  └──────────────────┘                      │
  │                                            │
  │  ┌──────────────────┐  Weighted Fair       │
  │  │ Video (CBWFQ)    │──────────────────►   │
  │  │ DSCP AF41        │  30% bandwidth       │
  │  └──────────────────┘                      │
  │                                            │
  │  ┌──────────────────┐                      │
  │  │ Critical (CBWFQ) │──────────────────►   │
  │  │ DSCP AF31        │  25% bandwidth       │
  │  └──────────────────┘                      │
  │                                            │
  │  ┌──────────────────┐                      │
  │  │ Best Effort      │──────────────────►   │
  │  │ DSCP 0           │  remaining BW        │
  │  └──────────────────┘                      │
  └────────────────────────────────────────────┘

  LLQ = CBWFQ + strict priority for voice.
  Voice is always sent first, but policed to prevent starvation.
```

---

## 6. DSCP and Per-Hop Behavior

### 6.1 DSCP Field

The DSCP (Differentiated Services Code Point) uses the 6 most significant bits of the IP ToS byte:

```
IP Header ToS Byte:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ D │ S │ C │ P │ 3 │ 2 │ 1 │ 0 │
│   │   │   │   │   │   │   │   │
│←──── DSCP (6 bits) ────→│ECN│
└───┴───┴───┴───┴───┴───┴───┴───┘
```

### 6.2 Standard Per-Hop Behaviors

| PHB | DSCP Value | Binary | Purpose |
|-----|-----------|--------|---------|
| Default (BE) | 0 | 000000 | Best-effort traffic |
| CS1 (Scavenger) | 8 | 001000 | Below best-effort (bulk) |
| AF11 | 10 | 001010 | Assured Forwarding class 1, low drop |
| AF21 | 18 | 010010 | Assured Forwarding class 2, low drop |
| AF31 | 26 | 011010 | Assured Forwarding class 3, low drop |
| AF41 | 34 | 100010 | Assured Forwarding class 4, low drop |
| EF | 46 | 101110 | Expedited Forwarding (voice) |
| CS6 | 48 | 110000 | Network control |

### 6.3 Assured Forwarding Matrix

```
AF classes have 3 drop precedences each:

         │ Low Drop (1) │ Medium Drop (2) │ High Drop (3)
─────────┼──────────────┼─────────────────┼──────────────
Class 1  │  AF11 (10)   │   AF12 (12)     │  AF13 (14)
Class 2  │  AF21 (18)   │   AF22 (20)     │  AF23 (22)
Class 3  │  AF31 (26)   │   AF32 (28)     │  AF33 (30)
Class 4  │  AF41 (34)   │   AF42 (36)     │  AF43 (38)

During congestion, high-drop packets are discarded first.
This allows WRED (Weighted Random Early Detection) to
selectively drop lower-priority traffic within a class.
```

---

## 7. QoS Design Patterns

### 7.1 Enterprise Campus QoS

```
Trust Boundary:
  Phones → trusted (mark EF)
  PCs    → untrusted (re-mark at access switch)

  Access switch:  classify + mark + police
  Distribution:   re-mark if needed
  Core:           honor markings, priority queue
  WAN edge:       shape to link speed, LLQ/CBWFQ
```

### 7.2 VoIP QoS Checklist

| Requirement | Configuration |
|-------------|--------------|
| Marking | DSCP EF (46) for voice RTP |
| Signaling | DSCP CS3 (24) for SIP/SCCP |
| Priority queue | LLQ with policing at 10-20% of link |
| Jitter buffer | 30-50ms at endpoints |
| Max latency | 150ms end-to-end |
| Bandwidth | ~100 Kbps per G.711 call |

---

## 8. Exercises

### Exercise 1: Token Bucket Simulator

Implement and test a single-rate token bucket:
1. CIR = 1 Mbps, CBS = 10 KB
2. Feed packets of varying sizes at different rates
3. Plot conform vs exceed decisions over time
4. Show how burst tolerance changes with CBS

### Exercise 2: WFQ Scheduler

Implement weighted fair queuing:
1. Create 4 queues with weights: voice=40, video=30, data=20, bulk=10
2. Generate packets for each queue
3. Schedule 100 time slots and count packets served per queue
4. Verify that bandwidth distribution matches weights

### Exercise 3: DiffServ Classifier

Build a traffic classifier:
1. Classify packets by 5-tuple (src/dst IP, ports, protocol)
2. Assign DSCP values based on classification rules
3. Implement a trust boundary: re-mark untrusted traffic
4. Count packets per DSCP class

### Exercise 4: WRED Simulation

Simulate Weighted Random Early Detection:
1. Implement a queue with min-threshold and max-threshold
2. When queue depth is between min and max, randomly drop with increasing probability
3. Higher drop precedence (AF13) has lower min-threshold than low drop (AF11)
4. Show that WRED reduces tail-drop synchronization

### Exercise 5: End-to-End QoS Analysis

Model a 3-hop network with QoS:
1. Each link: 10 Mbps, LLQ/CBWFQ configured
2. Generate mixed traffic: voice (EF), video (AF41), data (BE)
3. Measure per-hop delay, jitter, and loss for each class
4. Show that voice maintains < 150ms end-to-end even under congestion

---

*End of Lesson 21*
