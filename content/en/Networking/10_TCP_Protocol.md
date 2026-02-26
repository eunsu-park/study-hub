# TCP Protocol

**Previous**: [Routing Protocols](./09_Routing_Protocols.md) | **Next**: [UDP and Ports](./11_UDP_and_Ports.md)

**Difficulty**: ⭐⭐⭐
**Estimated Learning Time**: 3-4 hours
**Prerequisites**: [09_Routing_Protocols.md](./09_Routing_Protocols.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain TCP's core characteristics: connection-oriented, reliable, ordered delivery
2. Identify and describe the fields in a TCP header, including flags, sequence numbers, and window size
3. Trace the 3-way handshake process for connection establishment
4. Trace the 4-way handshake process for connection termination
5. Explain how sequence numbers and acknowledgments provide reliable delivery
6. Describe TCP flow control using the sliding window mechanism
7. Compare TCP congestion control algorithms: Slow Start, Congestion Avoidance, Fast Retransmit, and Fast Recovery

---

TCP is the protocol that makes the Internet reliable. Every time you download a file, load a web page, or send an email, TCP quietly ensures that every byte arrives intact, in order, and without duplication -- even across an unreliable network of routers and links. Understanding how TCP establishes connections, tracks data with sequence numbers, and adapts to network congestion is essential for debugging application performance issues and designing networked systems.

## Table of Contents

1. [TCP Characteristics](#1-tcp-characteristics)
2. [TCP Header Structure](#2-tcp-header-structure)
3. [3-Way Handshake](#3-3-way-handshake)
4. [4-Way Handshake](#4-4-way-handshake)
5. [Sequence Numbers and ACK](#5-sequence-numbers-and-ack)
6. [Flow Control](#6-flow-control)
7. [Congestion Control](#7-congestion-control)
8. [Practice Problems](#8-practice-problems)
9. [References](#9-references)

---

## 1. TCP Characteristics

### 1.1 Basic TCP Features

```
┌─────────────────────────────────────────────────────────────────┐
│                       TCP Features                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Connection-Oriented                                         │
│     - Requires connection establishment before data transfer    │
│       (3-way handshake)                                         │
│     - Connection termination after transmission (4-way handshake)│
│                                                                  │
│  2. Reliability                                                 │
│     - Guaranteed data delivery                                  │
│     - Order preservation                                        │
│     - Error detection and retransmission                        │
│                                                                  │
│  3. Flow Control                                                │
│     - Transmission rate matched to receiver's processing speed  │
│     - Uses sliding window                                       │
│                                                                  │
│  4. Congestion Control                                          │
│     - Responds to network congestion                            │
│     - Slow Start, Congestion Avoidance, etc.                   │
│                                                                  │
│  5. Full-Duplex Communication                                   │
│     - Simultaneous bidirectional data transmission              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 TCP vs UDP Brief Comparison

| Feature | TCP | UDP |
|------|-----|-----|
| Connection | Connection-oriented | Connectionless |
| Reliability | Reliable | Unreliable |
| Ordering | Ordered | Unordered |
| Speed | Relatively slow | Fast |
| Header Size | 20-60 bytes | 8 bytes |
| Use Cases | Web, email, file transfer | Streaming, DNS, gaming |

### 1.3 TCP Segment

```
TCP Data Encapsulation

┌─────────────────────────────────────────────────────────────────┐
│ Application Layer                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                         Data                                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Transport Layer (TCP)                                           │
│ ┌──────────────┬──────────────────────────────────────────────┐ │
│ │  TCP Header  │                    Data                       │ │
│ │   (20-60B)   │                (Segment)                      │ │
│ └──────────────┴──────────────────────────────────────────────┘ │
│                      TCP Segment                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Network Layer (IP)                                              │
│ ┌──────────────┬──────────────────────────────────────────────┐ │
│ │  IP Header   │              TCP Segment                      │ │
│ │   (20-60B)   │                                               │ │
│ └──────────────┴──────────────────────────────────────────────┘ │
│                       IP Packet                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. TCP Header Structure

### 2.1 TCP Header Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source Port          |       Destination Port        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Sequence Number                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Acknowledgment Number                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Data |       |C|E|U|A|P|R|S|F|                               |
| Offset| Rsrvd |W|C|R|C|S|S|Y|I|            Window             |
|       |       |R|E|G|K|H|T|N|N|                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Checksum            |         Urgent Pointer        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Options (if any)                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             Data                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### 2.2 Header Field Descriptions

| Field | Size | Description |
|------|------|------|
| Source Port | 16 bits | Source port number (0-65535) |
| Destination Port | 16 bits | Destination port number (0-65535) |
| Sequence Number | 32 bits | Byte number of first byte in segment data |
| Acknowledgment Number | 32 bits | Next expected byte number |
| Data Offset | 4 bits | TCP header length (in 4-byte units) |
| Reserved | 4 bits | Reserved (set to 0) |
| Flags | 8 bits | Control flags (CWR, ECE, URG, ACK, PSH, RST, SYN, FIN) |
| Window | 16 bits | Receive window size (flow control) |
| Checksum | 16 bits | Error detection checksum |
| Urgent Pointer | 16 bits | Urgent data location (when URG flag set) |
| Options | 0-40 bytes | Additional options (MSS, Window Scale, etc.) |

### 2.3 TCP Flags

```
┌─────────────────────────────────────────────────────────────────┐
│                       TCP Flags                                  │
├─────────┬───────────────────────────────────────────────────────┤
│ CWR     │ Congestion Window Reduced - congestion window reduced │
│ ECE     │ ECN-Echo - explicit congestion notification           │
│ URG     │ Urgent - urgent data present                          │
│ ACK     │ Acknowledgment - acknowledgment valid                 │
│ PSH     │ Push - deliver immediately without buffering          │
│ RST     │ Reset - force connection termination                  │
│ SYN     │ Synchronize - connection request (seq number sync)    │
│ FIN     │ Finish - connection termination request               │
└─────────┴───────────────────────────────────────────────────────┘
```

### 2.4 Key TCP Options

| Option | Kind | Description |
|------|------|------|
| MSS | 2 | Maximum Segment Size (typically 1460 bytes) |
| Window Scale | 3 | Window size expansion (up to 1GB) |
| SACK Permitted | 4 | Selective ACK support |
| SACK | 5 | Received segment ranges |
| Timestamps | 8 | RTT measurement and PAWS |
| NOP | 1 | Padding (No Operation) |

---

> **Analogy -- Registered Mail**: TCP is like sending a registered letter. You get a delivery confirmation (ACK), the postal service tracks every package (sequence numbers), and if something gets lost, it's automatically re-sent (retransmission). This reliability comes at a cost -- it's slower than just dropping a postcard in the mailbox (UDP). But when you need to be certain your message arrived intact and in order, TCP's overhead is worth the guarantee.

## 3. 3-Way Handshake

### 3.1 Connection Establishment Process

TCP connections are established through a 3-way handshake.

```
┌─────────────────────────────────────────────────────────────────┐
│                    3-Way Handshake                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│     Client                               Server                 │
│         │                                  │                    │
│         │  Step 1: SYN                     │                    │
│         │  ─────────────────────────────►  │                    │
│         │  SYN=1, Seq=100                  │                    │
│  CLOSED │                                  │ LISTEN             │
│    ↓    │                                  │    ↓               │
│ SYN_SENT│  Step 2: SYN-ACK                 │ SYN_RECEIVED       │
│         │  ◄─────────────────────────────  │                    │
│         │  SYN=1, ACK=1, Seq=300, Ack=101  │                    │
│    ↓    │                                  │                    │
│ ESTABLISHED Step 3: ACK                    │                    │
│         │  ─────────────────────────────►  │                    │
│         │  ACK=1, Seq=101, Ack=301         │                    │
│         │                                  │ ESTABLISHED        │
│         │                                  │                    │
│         │        Connection Established    │                    │
│         │  ◄═════════════════════════════► │                    │
│         │         Data Transfer            │                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Detailed Steps

**Step 1: SYN (Synchronize)**
```
Client → Server

TCP Header:
  Source Port: 50000 (ephemeral)
  Destination Port: 80 (HTTP)
  Sequence Number: 100 (ISN - Initial Sequence Number)
  Acknowledgment: 0
  Flags: SYN=1
  Window: 65535
  Options: MSS=1460, Window Scale=7

Meaning: "I want to connect. My sequence number starts at 100."
```

**Step 2: SYN-ACK**
```
Server → Client

TCP Header:
  Source Port: 80
  Destination Port: 50000
  Sequence Number: 300 (Server's ISN)
  Acknowledgment: 101 (Client Seq + 1)
  Flags: SYN=1, ACK=1
  Window: 65535
  Options: MSS=1460, Window Scale=7

Meaning: "Connection request received. Expecting byte 101.
         My sequence number starts at 300."
```

**Step 3: ACK**
```
Client → Server

TCP Header:
  Source Port: 50000
  Destination Port: 80
  Sequence Number: 101
  Acknowledgment: 301 (Server Seq + 1)
  Flags: ACK=1
  Window: 65535

Meaning: "Received server's response. Expecting byte 301.
         Now we can exchange data."
```

### 3.3 ISN (Initial Sequence Number)

```
Why ISN is Random:

1. Security
   - Predictable ISN vulnerable to TCP session hijacking
   - Random ISN increases attack difficulty

2. Distinguish from Previous Connections
   - Prevents confusion with packets from previous connection
     on same socket (IP:Port pair)
   - Related to TIME_WAIT state

ISN Generation Example:
  Modern OS: Uses secure random number generator (CSPRNG)
  Legacy: Time-based counter (increment by 1 every 4 microseconds)
```

### 3.4 TCP State Transition (Connection Establishment)

```
Client state transition:
CLOSED → SYN_SENT → ESTABLISHED

Server state transition:
CLOSED → LISTEN → SYN_RECEIVED → ESTABLISHED
```

---

## 4. 4-Way Handshake

### 4.1 Connection Termination Process

TCP connections are gracefully terminated through a 4-way handshake.

```
┌─────────────────────────────────────────────────────────────────┐
│                    4-Way Handshake                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Client (Active Close)                Server (Passive Close)   │
│         │                                  │                    │
│ ESTABLISHED                          ESTABLISHED               │
│         │                                  │                    │
│         │  Step 1: FIN                     │                    │
│         │  ─────────────────────────────►  │                    │
│         │  FIN=1, Seq=100                  │                    │
│ FIN_WAIT_1                                 │                    │
│         │                                  │ CLOSE_WAIT         │
│         │  Step 2: ACK                     │                    │
│         │  ◄─────────────────────────────  │                    │
│         │  ACK=1, Ack=101                  │                    │
│ FIN_WAIT_2                                 │                    │
│         │                                  │ (Send remaining data)│
│         │                                  │                    │
│         │  Step 3: FIN                     │                    │
│         │  ◄─────────────────────────────  │                    │
│         │  FIN=1, Seq=300                  │ LAST_ACK           │
│ TIME_WAIT                                  │                    │
│         │  Step 4: ACK                     │                    │
│         │  ─────────────────────────────►  │                    │
│         │  ACK=1, Ack=301                  │                    │
│         │                                  │ CLOSED             │
│  (Wait 2MSL)                               │                    │
│ CLOSED                                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Detailed Steps

| Step | Sender | Flags | Description |
|------|--------|--------|------|
| 1 | Client | FIN | "No more data to send. Request connection termination" |
| 2 | Server | ACK | "FIN acknowledged. May still have data to send" |
| 3 | Server | FIN | "I'm done sending too. Agree to terminate" |
| 4 | Client | ACK | "FIN acknowledged. Connection terminated" |

### 4.3 Half-Close

TCP supports half-close. Even after one side sends FIN, the other can continue sending data.

```
Half-Close Scenario

Client              Server
    │                  │
    │── FIN ─────────►│  Client: "No more data to send"
    │                  │
    │◄──── ACK ───────│  Server: "OK"
    │                  │
    │◄──── Data ──────│  Server: Continue sending remaining data
    │◄──── Data ──────│
    │                  │
    │── ACK ─────────►│
    │                  │
    │◄──── FIN ───────│  Server: "I'm done too"
    │                  │
    │── ACK ─────────►│  Connection closed
    │                  │
```

### 4.4 TIME_WAIT State

```
Purpose of TIME_WAIT:

1. Handle Delayed Packets
   - Wait for old connection packets in network to expire
   - Prevent confusion with new connection

2. Handle Lost Final ACK
   - If server doesn't receive final ACK, it will resend FIN
   - Client in TIME_WAIT can respond again

TIME_WAIT Duration: 2 × MSL (Maximum Segment Lifetime)
  - MSL: Typically 30 seconds or 2 minutes
  - TIME_WAIT: 1 minute ~ 4 minutes

┌─────────────────────────────────────────────────────────────────┐
│  TIME_WAIT Problem                                              │
├─────────────────────────────────────────────────────────────────┤
│  - Can cause port exhaustion with many short-lived connections  │
│  - Solutions:                                                   │
│    1. Use SO_REUSEADDR socket option                            │
│    2. tcp_tw_reuse kernel parameter (Linux)                     │
│    3. Use connection pooling                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 TCP State Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                 TCP State Diagram (Simplified)                   │
└─────────────────────────────────────────────────────────────────┘

                              CLOSED
                                │
              ┌─────────────────┼─────────────────┐
              │ Active open     │ Passive open    │
              ▼                 ▼                 │
          SYN_SENT ──────► LISTEN                │
              │                 │                 │
              │  Receive SYN    │ Receive SYN     │
              ▼                 ▼                 │
          ◄──────────── SYN_RCVD ─────────►      │
                              │                   │
                    Receive   │                   │
                    ACK       ▼                   │
                        ESTABLISHED               │
                              │                   │
              ┌───────────────┴───────────────┐   │
              │ Active close   Passive close  │   │
              ▼                               ▼   │
          FIN_WAIT_1                    CLOSE_WAIT │
              │                               │   │
              ▼                               ▼   │
          FIN_WAIT_2                     LAST_ACK │
              │                               │   │
              ▼                               │   │
          TIME_WAIT ─────────────────────────►│   │
              │                                   │
              └─── 2MSL ──────────────────────────┘
                              │
                              ▼
                           CLOSED
```

---

## 5. Sequence Numbers and ACK

### 5.1 Role of Sequence Numbers

```
Sequence Number Operation

Client                               Server
    │                                      │
    │── Seq=1000, 1000 bytes ────────────►│
    │       (1000-1999)                    │
    │                                      │
    │◄─────────────────── ACK=2000 ───────│
    │   "Please send from byte 2000"       │
    │                                      │
    │── Seq=2000, 1000 bytes ────────────►│
    │       (2000-2999)                    │
    │                                      │
    │◄─────────────────── ACK=3000 ───────│
    │                                      │

Sequence number = Byte number of first data byte in segment
ACK number = Next expected byte number
```

### 5.2 Cumulative ACK

```
Cumulative ACK Operation

Sender                                   Receiver
    │                                      │
    │── Seq=1000, 500 bytes ─────────────►│
    │── Seq=1500, 500 bytes ─────────────►│
    │── Seq=2000, 500 bytes ─────────────►│
    │                                      │
    │◄────────────────────── ACK=2500 ────│
    │                                      │
    │  Single ACK acknowledges all 3 segments │
    │                                      │

Advantage:
- Reduces ACK packet count
- Improves network efficiency

Disadvantage:
- Lost middle packet requires retransmission of subsequent packets
  (Solved by SACK)
```

### 5.3 SACK (Selective Acknowledgment)

```
SACK Operation

Sender                                   Receiver
    │                                      │
    │── Seq=1000, 500B ─────────────────►│ ✓
    │── Seq=1500, 500B ────────────X     │ (lost)
    │── Seq=2000, 500B ─────────────────►│ ✓
    │── Seq=2500, 500B ─────────────────►│ ✓
    │                                      │
    │◄──── ACK=1500, SACK=2000-3000 ─────│
    │      "1500 missing, but received 2000-3000"
    │                                      │
    │── Seq=1500, 500B ─────────────────►│ (retransmit)
    │                                      │
    │◄────────────────────── ACK=3000 ────│
    │                                      │

SACK Advantages:
- Selective retransmission of lost segments only
- Prevents unnecessary retransmissions
- Efficient on high-speed networks
```

### 5.4 Retransmission Timer (RTO)

```
RTO (Retransmission Timeout) Calculation

1. Measure RTT (Round Trip Time)
   - Time from segment transmission to ACK reception

2. Calculate SRTT (Smoothed RTT)
   SRTT = (1 - α) × SRTT + α × RTT
   (α = 1/8, typically)

3. Calculate RTTVAR (RTT Variance)
   RTTVAR = (1 - β) × RTTVAR + β × |SRTT - RTT|
   (β = 1/4, typically)

4. Calculate RTO
   RTO = SRTT + max(G, 4 × RTTVAR)
   (G = clock granularity)

Example:
  Measured RTT: 100ms, 120ms, 90ms, 110ms
  SRTT ≈ 105ms
  RTTVAR ≈ 10ms
  RTO = 105 + 4 × 10 = 145ms
```

---

## 6. Flow Control

### 6.1 Sliding Window

```
Sliding Window Concept

Send Buffer:
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
│ACKed│    Window (can send)      │   Cannot send yet│
│     │◄────── Window=5 ───────►│                    │

Control transmission volume based on receive window size:

Window = 5000:
┌──────────────────────────────────────────────────┐
│     Can send up to 5000 bytes without ACK        │
└──────────────────────────────────────────────────┘

Window = 0 (Zero Window):
┌──────────────────────────────────────────────────┐
│     Stop transmission! Receiver buffer full      │
│     Periodic check with Window Probe            │
└──────────────────────────────────────────────────┘
```

### 6.2 Window Size Adjustment

```
Window Size Changes

Sender                               Receiver
    │                                  │ rcvbuf=4000
    │── 1000B ────────────────────────►│ Window=3000
    │── 1000B ────────────────────────►│ Window=2000
    │── 1000B ────────────────────────►│ Window=1000
    │                                  │
    │◄─── ACK, Window=0 (buffer full) ─│
    │                                  │
    │     (Application reads data)      │
    │                                  │
    │◄─── ACK, Window=4000 ────────────│
    │                                  │
    │── 1000B ────────────────────────►│
    │── 1000B ────────────────────────►│
```

### 6.3 Window Scale Option

```
Default window size: 16 bits = max 65,535 bytes

Insufficient for high-speed networks:
  - 100Mbps, RTT 100ms → BDP = 100 × 0.1 / 8 = 1.25MB needed
  - 1Gbps, RTT 100ms → BDP = 12.5MB needed

Window Scale Option:
  - Negotiated during 3-way handshake
  - Left-shifts window value
  - Maximum scale: 14 (window max ~1GB)

Example:
  Window = 65535, Scale = 7
  Actual window = 65535 × 2^7 = 65535 × 128 = 8,388,480 bytes (~8MB)
```

### 6.4 Silly Window Syndrome Prevention

```
Problem: Transmitting many small segments (inefficient)

Sender Solution: Nagle's Algorithm
┌─────────────────────────────────────────────────────────────────┐
│  When there's small data:                                       │
│  1. If no outstanding data, send immediately                    │
│  2. If outstanding data exists, wait until MSS accumulated      │
│     or ACK received                                             │
└─────────────────────────────────────────────────────────────────┘

Receiver Solution: Delayed ACK + Clark's Solution
┌─────────────────────────────────────────────────────────────────┐
│  1. Don't send ACK immediately, wait 200ms                      │
│  2. Window update only when MSS or 50% of buffer available      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Congestion Control

### 7.1 Congestion Control Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TCP Congestion Control                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Purpose: Detect network congestion and adjust transmission rate│
│                                                                  │
│  Key Variables:                                                 │
│  - cwnd (Congestion Window): Sender-determined window           │
│  - rwnd (Receive Window): Receiver-advertised window            │
│  - Actual transmission = min(cwnd, rwnd)                        │
│                                                                  │
│  ssthresh (Slow Start Threshold):                               │
│  - Boundary between Slow Start and Congestion Avoidance         │
│  - Adjusted on congestion                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Slow Start

```
Slow Start Operation

                cwnd Change
               │
       64 MSS  ┤                              * (congestion)
               │                            *
       32 MSS  ┤                          *
               │                        *
       16 MSS  ┤                      *
               │                    *
        8 MSS  ┤                  *
               │                *
        4 MSS  ┤              *
               │            *
        2 MSS  ┤          *
               │        *
        1 MSS  ┤      *
               │
               └──────────────────────────────────── RTT
                    1   2   3   4   5   6

Rules:
- Initial cwnd = 1 MSS (or IW=10 MSS, modern implementations)
- For each ACK, cwnd += 1 MSS
- Result: cwnd doubles every RTT (exponential growth)
- Switch to Congestion Avoidance when ssthresh reached
```

### 7.3 Congestion Avoidance

```
Congestion Avoidance Operation

                cwnd Change
               │
               │  ssthresh                    * (congestion)
        16 MSS ┼─────────────*──────────*──*
               │           *   *       *
               │         *      *    *
               │       *          *
               │     *       Linear increase (AIMD)
               │   *
               │ *  Slow Start (exponential)
               │*
               │
               └──────────────────────────────────── RTT

Rules:
- Operates when cwnd >= ssthresh
- For each RTT, cwnd += 1 MSS (or cwnd += MSS/cwnd per ACK)
- Linear increase (Additive Increase)
```

### 7.4 Congestion Detection and Response

```
Congestion Detection Methods:

1. Timeout (RTO expiration)
   - Judged as severe congestion
   - ssthresh = cwnd / 2
   - cwnd = 1 MSS
   - Restart Slow Start

2. 3 Duplicate ACKs (Fast Retransmit)
   - Judged as mild congestion
   - ssthresh = cwnd / 2
   - cwnd = ssthresh + 3 MSS
   - Enter Fast Recovery

┌─────────────────────────────────────────────────────────────────┐
│                     Congestion Control State Transition          │
│                                                                  │
│  Slow Start ──(cwnd >= ssthresh)──► Congestion Avoidance        │
│      │                                        │                  │
│      │                                        │                  │
│   (timeout)                              (3 dup ACKs)            │
│      │                                        │                  │
│      ▼                                        ▼                  │
│  Slow Start ◄──────────────────────── Fast Recovery             │
│                    (recovery complete)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 7.5 Fast Retransmit and Fast Recovery

```
Fast Retransmit Scenario

Sender                               Receiver
    │                                  │
    │── Seq=1000 ─────────────────────►│
    │── Seq=2000 ──────X              │ (lost)
    │── Seq=3000 ─────────────────────►│
    │                                  │
    │◄───────────────── ACK=2000 (dup1)│
    │                                  │
    │── Seq=4000 ─────────────────────►│
    │◄───────────────── ACK=2000 (dup2)│
    │                                  │
    │── Seq=5000 ─────────────────────►│
    │◄───────────────── ACK=2000 (dup3)│
    │                                  │
    │  Received 3 duplicate ACKs!      │
    │  → Retransmit immediately        │
    │    without waiting for RTO       │
    │                                  │
    │── Seq=2000 (retransmit) ────────►│
    │                                  │
    │◄─────────────────────── ACK=6000 │

Fast Recovery:
- After 3 dup ACKs, ssthresh = cwnd/2
- cwnd = ssthresh + 3 (account for received segments)
- On new ACK, cwnd = ssthresh
- Switch to Congestion Avoidance
```

### 7.6 Modern Congestion Control Algorithms

| Algorithm | Features | Environment |
|----------|------|----------|
| Reno | Basic AIMD, Fast Recovery | Standard |
| NewReno | Improved partial ACK handling | Reno improvement |
| CUBIC | BIC improvement, Linux default | High-speed networks |
| BBR | Bandwidth/RTT based | Google, high latency networks |
| Vegas | RTT change based | Low latency environments |

```
CUBIC cwnd Growth

cwnd
  │
  │                              *
  │                           *     *
  │                        *           *
  │                     *                 *
  │                  *                       *
  │               *                             *
  │            *          cubic function          *
  │         *                                        *
  │      *
  │   *
  │ *
  │*
  └──────────────────────────────────────────────────── time

Features:
- Remembers W_max (window at last congestion)
- Fast approach to W_max, then slow growth
- Excellent fairness and scalability
```

---

## 8. Practice Problems

### Problem 1: 3-Way Handshake Analysis

Analyze the following packet capture.

```
Packet 1: 192.168.1.10:50000 → 10.0.0.5:443
          SYN, Seq=1000000000

Packet 2: 10.0.0.5:443 → 192.168.1.10:50000
          SYN, ACK, Seq=2000000000, Ack=?

Packet 3: 192.168.1.10:50000 → 10.0.0.5:443
          ACK, Seq=?, Ack=?
```

a) What is the Ack value in Packet 2?
b) What is the Seq value in Packet 3?
c) What is the Ack value in Packet 3?

### Problem 2: Sequence Number Calculation

Client sends 5000 bytes to server. MSS=1000 bytes.

Calculate Seq number and expected ACK for each segment when initial sequence number is 10000.

| Segment | Data Size | Seq | Expected ACK |
|----------|------------|-----|----------|
| 1        | 1000       |     |          |
| 2        | 1000       |     |          |
| 3        | 1000       |     |          |
| 4        | 1000       |     |          |
| 5        | 1000       |     |          |

### Problem 3: Flow Control

Receiver's receive buffer is 10000 bytes. Currently 2000 bytes in buffer.

a) What is the advertised window size?
b) If sender transmits 4000 bytes, what's the new window size?
c) If application reads 3000 bytes, what's the new window size?

### Problem 4: Congestion Control

Starting with ssthresh = 16 MSS, cwnd = 1 MSS.

a) What is cwnd size after 4 RTTs? (no loss)
b) When cwnd = 32 MSS, timeout occurs. What are new ssthresh and cwnd?
c) When cwnd = 24 MSS, 3 duplicate ACKs occur. What are new ssthresh and cwnd?

---

## Answers

### Problem 1 Answers

a) Packet 2 Ack = **1000000001** (Client Seq + 1)
b) Packet 3 Seq = **1000000001** (SYN counts as 1 byte)
c) Packet 3 Ack = **2000000001** (Server Seq + 1)

### Problem 2 Answers

| Segment | Data Size | Seq | Expected ACK |
|----------|------------|-----|----------|
| 1        | 1000       | 10000 | 11000 |
| 2        | 1000       | 11000 | 12000 |
| 3        | 1000       | 12000 | 13000 |
| 4        | 1000       | 13000 | 14000 |
| 5        | 1000       | 14000 | 15000 |

### Problem 3 Answers

a) Advertised window = 10000 - 2000 = **8000 bytes**
b) New window = 10000 - 2000 - 4000 = **4000 bytes**
c) New window = 10000 - (2000 + 4000 - 3000) = **7000 bytes**

### Problem 4 Answers

a) Slow Start phase (cwnd < ssthresh)
   - RTT 1: cwnd = 2 MSS
   - RTT 2: cwnd = 4 MSS
   - RTT 3: cwnd = 8 MSS
   - RTT 4: cwnd = **16 MSS**

b) Timeout occurs:
   - New ssthresh = 32 / 2 = **16 MSS**
   - New cwnd = **1 MSS**

c) 3 dup ACKs (Fast Retransmit):
   - New ssthresh = 24 / 2 = **12 MSS**
   - New cwnd = 12 + 3 = **15 MSS** (Fast Recovery)

---

**Previous**: [Routing Protocols](./09_Routing_Protocols.md) | **Next**: [UDP and Ports](./11_UDP_and_Ports.md)

---

## 9. References

### RFC Documents

- RFC 793 - TCP Basic Specification
- RFC 5681 - TCP Congestion Control
- RFC 7323 - TCP Extensions (Window Scaling, Timestamps)
- RFC 2018 - TCP Selective Acknowledgment Options

### Command Reference

```bash
# Check TCP connections (Linux)
ss -tan
netstat -an | grep tcp

# TCP statistics
netstat -s | grep -i tcp
cat /proc/net/snmp | grep Tcp

# Check TCP tuning parameters
sysctl net.ipv4.tcp_congestion_control
sysctl net.core.rmem_max
sysctl net.ipv4.tcp_window_scaling

# Wireshark filters
tcp.flags.syn == 1 && tcp.flags.ack == 0  # SYN packets
tcp.analysis.retransmission               # Retransmissions
tcp.analysis.duplicate_ack                # Duplicate ACKs
```

### Learning Resources

- [TCP/IP Illustrated, Vol. 1 - W. Richard Stevens](https://www.amazon.com/TCP-Illustrated-Vol-Addison-Wesley-Professional/dp/0201633469)
- [High Performance Browser Networking](https://hpbn.co/)
- [Cloudflare Blog - TCP](https://blog.cloudflare.com/tag/tcp/)

---

**Document Information**
- Last Updated: 2024
- Difficulty: ⭐⭐⭐
- Estimated Learning Time: 3-4 hours
