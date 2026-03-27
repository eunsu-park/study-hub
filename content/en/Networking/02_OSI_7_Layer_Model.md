# OSI 7-Layer Model

**Previous**: [Network Fundamentals](./01_Network_Fundamentals.md) | **Next**: [TCP/IP Model](./03_TCP_IP_Model.md)

**Difficulty**: ⭐⭐ (Beginner-Intermediate)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of the OSI reference model and why layered architecture matters
2. Describe the function of each of the seven OSI layers
3. Identify the PDU (Protocol Data Unit) name at each layer
4. Trace the encapsulation and decapsulation process as data moves between layers
5. Map common protocols and network devices to their corresponding OSI layers
6. Apply the OSI model as a framework for systematic network troubleshooting

---

The OSI model is the "Rosetta Stone" of networking -- it gives engineers, developers, and students a shared vocabulary for describing how data travels from one application to another across a network. Even though the Internet runs on the simpler TCP/IP stack, the OSI model remains the standard mental framework used in every network certification, troubleshooting session, and design discussion.

> **Analogy -- The Postal System**: The OSI model works like a postal system with seven departments. When you send a letter, the Application layer is you writing the letter, the Presentation layer is putting it in an envelope, the Session layer is the conversation thread, the Transport layer is choosing registered vs. regular mail, the Network layer is the postal routing system deciding which sorting centers to use, the Data Link layer is the local mail carrier on your street, and the Physical layer is the actual truck carrying the mailbag. Each layer only talks to the layer directly above and below it -- just as a mail carrier doesn't need to know what's written in the letter.

## Table of Contents

1. [OSI Model Overview](#1-osi-model-overview)
2. [Detailed Explanation of 7 Layers](#2-detailed-explanation-of-7-layers)
3. [Protocols by Layer](#3-protocols-by-layer)
4. [PDU (Protocol Data Unit)](#4-pdu-protocol-data-unit)
5. [Encapsulation and Decapsulation](#5-encapsulation-and-decapsulation)
6. [Key Devices by Layer](#6-key-devices-by-layer)
7. [Practical Application of OSI Model](#7-practical-application-of-osi-model)
8. [Practice Problems](#8-practice-problems)

---

## 1. OSI Model Overview

### What is the OSI Model?

```
┌─────────────────────────────────────────────────────────────────┐
│                 OSI (Open Systems Interconnection)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   "Reference model for open systems interconnection"            │
│                                                                  │
│   Purpose:                                                      │
│   1. Standardize network communication processes                │
│   2. Ensure compatibility between different vendor equipment    │
│   3. Systematic approach to network troubleshooting             │
│   4. Provide common language for developers and engineers       │
│                                                                  │
│   History:                                                      │
│   - 1977: ISO begins work                                       │
│   - 1984: OSI reference model published (ISO 7498)              │
│   - Present: Used as educational/reference model                │
│             rather than actual implementation                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7-Layer Structure Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      OSI 7-Layer Model                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Layer Number     Layer Name           Main Function           │
│   ─────────────────────────────────────────────────────────────│
│                                                                  │
│     ┌─────────────────────────────────────────────────────┐     │
│   7 │             Application Layer                        │     │
│     │        Interface between user and network            │     │
│     ├─────────────────────────────────────────────────────┤     │
│   6 │             Presentation Layer                       │     │
│     │        Data format conversion, encryption,           │     │
│     │        compression                                   │     │
│     ├─────────────────────────────────────────────────────┤     │
│   5 │              Session Layer                           │     │
│     │        Manage connection setup, maintenance,         │     │
│     │        termination                                   │     │
│     ├─────────────────────────────────────────────────────┤     │
│   4 │              Transport Layer                         │     │
│     │        End-to-end reliable data transmission         │     │
│     ├─────────────────────────────────────────────────────┤     │
│   3 │            Network Layer                             │     │
│     │        Logical addressing, routing                   │     │
│     ├─────────────────────────────────────────────────────┤     │
│   2 │           Data Link Layer                            │     │
│     │        Physical addressing, frame transmission       │     │
│     ├─────────────────────────────────────────────────────┤     │
│   1 │              Physical Layer                          │     │
│     │        Bit transmission, physical connection         │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                  │
│   Mnemonic (top to bottom):                                     │
│   "All People Seem To Need Data Processing"                     │
│   (Application, Presentation, Session, Transport,               │
│    Network, Data Link, Physical)                                │
│                                                                  │
│   Mnemonic (bottom to top):                                     │
│   "Please Do Not Throw Sausage Pizza Away"                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                       Layer Classification                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Upper Layers (Host Layers) - Software implementation         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  7. Application  │                                       │   │
│   │  6. Presentation │  → Data processing/presentation       │   │
│   │  5. Session      │  → Application support                │   │
│   │  4. Transport    │                                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Lower Layers (Media Layers) - Hardware/firmware              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  3. Network      │                                       │   │
│   │  2. Data Link    │  → Actual data transmission           │   │
│   │  1. Physical     │  → Network infrastructure             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Explanation of 7 Layers

### Layer 7: Application Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                  Layer 7: Application                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Provides interface between user and network services    │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                      User                                │   │
│   │                        │                                 │   │
│   │                   [Web Browser]                          │   │
│   │                        │                                 │   │
│   │                   ┌────┴────┐                            │   │
│   │                   │Application│                          │   │
│   │                   │  (HTTP)   │                          │   │
│   │                   └─────────┘                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Functions:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. Provide network services                             │   │
│   │    - File transfer, email, web browsing                 │   │
│   │                                                          │   │
│   │ 2. Implement application protocols                      │   │
│   │    - HTTP, FTP, SMTP, DNS, SSH                          │   │
│   │                                                          │   │
│   │ 3. Generate and display data                            │   │
│   │    - Process user input, display results                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Protocol Examples:                                            │
│   ┌──────────┬───────────────────────────────────────────┐      │
│   │ HTTP/S   │ Web page transmission (port 80/443)       │      │
│   │ FTP      │ File transfer (port 20/21)                │      │
│   │ SMTP     │ Email transmission (port 25)              │      │
│   │ POP3     │ Email retrieval (port 110)                │      │
│   │ IMAP     │ Email access (port 143)                   │      │
│   │ DNS      │ Domain name resolution (port 53)          │      │
│   │ SSH      │ Remote access (port 22)                   │      │
│   │ Telnet   │ Remote access (port 23, unencrypted)      │      │
│   └──────────┴───────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 6: Presentation Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                 Layer 6: Presentation                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Data format conversion, encryption, compression         │
│         "Data translator"                                       │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Application Data                                      │   │
│   │         │                                                │   │
│   │         ▼                                                │   │
│   │   ┌─────────────────────────────────────────────┐       │   │
│   │   │              Presentation Layer              │       │   │
│   │   │                                              │       │   │
│   │   │   ┌──────────┐ ┌──────────┐ ┌──────────┐   │       │   │
│   │   │   │  Format  │ │Encryption│ │Compress. │   │       │   │
│   │   │   │Conversion│ │(SSL/TLS) │ │  (GZIP)  │   │       │   │
│   │   │   │  (Codec) │ │          │ │          │   │       │   │
│   │   │   └──────────┘ └──────────┘ └──────────┘   │       │   │
│   │   │                                              │       │   │
│   │   └─────────────────────────────────────────────┘       │   │
│   │         │                                                │   │
│   │         ▼                                                │   │
│   │   Transmittable Form                                    │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Functions:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. Data Translation                                     │   │
│   │    - Character encoding: ASCII, UTF-8, EBCDIC           │   │
│   │    - Data formats: JPEG, GIF, MPEG, HTML                │   │
│   │                                                          │   │
│   │ 2. Encryption/Decryption                                │   │
│   │    - SSL/TLS encryption                                 │   │
│   │    - Data security                                      │   │
│   │                                                          │   │
│   │ 3. Compression/Decompression                            │   │
│   │    - Reduce data size                                   │   │
│   │    - Improve transmission efficiency                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Examples:                                                     │
│   - SSL/TLS encryption in HTTPS                                │
│   - JPEG image compression                                     │
│   - Video streaming codecs (H.264, H.265)                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 5: Session Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                   Layer 5: Session                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Establish, maintain, and terminate communication        │
│         sessions between two systems                            │
│         "Conversation manager"                                  │
│                                                                  │
│   Session Lifecycle:                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │     ┌──────────┐                                        │   │
│   │     │Connection│  ← Session start, authentication       │   │
│   │     │ Setup    │                                        │   │
│   │     └────┬─────┘                                        │   │
│   │          │                                               │   │
│   │          ▼                                               │   │
│   │     ┌──────────┐                                        │   │
│   │     │  Data    │  ← Bidirectional communication         │   │
│   │     │Transfer  │  ← Set synchronization points          │   │
│   │     │          │    (checkpoints)                       │   │
│   │     └────┬─────┘                                        │   │
│   │          │                                               │   │
│   │          ▼                                               │   │
│   │     ┌──────────┐                                        │   │
│   │     │Connection│  ← Session termination                 │   │
│   │     │Teardown  │                                        │   │
│   │     └──────────┘                                        │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Functions:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. Dialog Control                                       │   │
│   │    - Simplex (one-way)                                  │   │
│   │    - Half-duplex                                        │   │
│   │    - Full-duplex                                        │   │
│   │                                                          │   │
│   │ 2. Synchronization                                      │   │
│   │    - Set checkpoints                                    │   │
│   │    - Provide recovery points on failure                 │   │
│   │                                                          │   │
│   │ 3. Session Management                                   │   │
│   │    - Authentication and authorization                   │   │
│   │    - Session ID management                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Communication Modes:                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Simplex:           A ───────────► B                   │   │
│   │                      (Radio broadcast)                   │   │
│   │                                                          │   │
│   │   Half-duplex:       A ◄─────────► B                   │   │
│   │                      (One direction at a time)          │   │
│   │                      (Walkie-talkie)                     │   │
│   │                                                          │   │
│   │   Full-duplex:       A ◄═════════► B                   │   │
│   │                      (Simultaneous bidirectional)       │   │
│   │                      (Telephone)                         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Protocols/Technologies: NetBIOS, RPC, PPTP, SIP              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 4: Transport Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                  Layer 4: Transport                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: End-to-end reliable data transmission                   │
│         "Quality manager of data transmission"                  │
│                                                                  │
│   End-to-End Communication:                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Sending Host                    Receiving Host        │   │
│   │   ┌─────────┐                     ┌─────────┐           │   │
│   │   │Application│                   │Application│         │   │
│   │   │ Process   │                   │ Process   │         │   │
│   │   └────┬────┘                     └────┬────┘           │   │
│   │        │                               │                │   │
│   │   ┌────┴────┐  ←─────────────────→ ┌────┴────┐         │   │
│   │   │Transport│    End-to-end         │Transport│         │   │
│   │   │  (TCP)  │    connection         │  (TCP)  │         │   │
│   │   │         │    (virtual)          │         │         │   │
│   │   └─────────┘                      └─────────┘         │   │
│   │        │           Network             │                │   │
│   │        └───────────────────────────────┘                │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Functions:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. Segmentation                                         │   │
│   │    - Divide large data into small segments              │   │
│   │    - Assign sequence numbers to each segment            │   │
│   │                                                          │   │
│   │ 2. Flow Control                                         │   │
│   │    - Adjust transmission speed to receiver's capacity   │   │
│   │    - Sliding window method                              │   │
│   │                                                          │   │
│   │ 3. Error Control                                        │   │
│   │    - Retransmit lost segments                           │   │
│   │    - Remove duplicate data                              │   │
│   │                                                          │   │
│   │ 4. Connection Management                                │   │
│   │    - Connection setup (3-way handshake)                 │   │
│   │    - Connection teardown (4-way handshake)              │   │
│   │                                                          │   │
│   │ 5. Process identification via port numbers              │   │
│   │    - Source/destination port numbers                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Major Protocols:                                              │
│   ┌─────────────┬──────────────────────────────────────────┐    │
│   │    TCP      │ Connection-oriented, reliability         │    │
│   │             │ Order guaranteed, flow/error control     │    │
│   │             │ Web, email, file transfer                │    │
│   ├─────────────┼──────────────────────────────────────────┤    │
│   │    UDP      │ Connectionless, unreliable               │    │
│   │             │ Fast transmission, low overhead          │    │
│   │             │ Streaming, DNS, games                    │    │
│   └─────────────┴──────────────────────────────────────────┘    │
│                                                                  │
│   Port Number Ranges:                                           │
│   - Well-known ports: 0-1023 (system/standard services)        │
│   - Registered ports: 1024-49151 (registered services)         │
│   - Dynamic ports: 49152-65535 (ephemeral/client)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 3: Network Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                 Layer 3: Network                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Logical addressing and optimal path determination       │
│         (routing)                                               │
│         "Delivery route designer of postal system"              │
│                                                                  │
│   Routing Concept:                                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Network A          Network B          Network C       │   │
│   │   ┌─────────┐        ┌─────────┐       ┌─────────┐     │   │
│   │   │192.168. │        │10.0.0.  │       │172.16.  │     │   │
│   │   │ 1.0/24  │        │0.0/8    │       │0.0/16   │     │   │
│   │   └────┬────┘        └────┬────┘       └────┬────┘     │   │
│   │        │                  │                  │          │   │
│   │        └───────┬──────────┴─────────┬────────┘          │   │
│   │                │                    │                    │   │
│   │           ┌────┴────┐          ┌────┴────┐              │   │
│   │           │ Router1 │──────────│ Router2 │              │   │
│   │           └─────────┘          └─────────┘              │   │
│   │                │                    │                    │   │
│   │                └──────────┬─────────┘                    │   │
│   │                           │                              │   │
│   │                      ┌────┴────┐                        │   │
│   │                      │Internet │                        │   │
│   │                      └─────────┘                        │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Functions:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. Logical Addressing                                   │   │
│   │    - IP address assignment                              │   │
│   │    - Network identification                             │   │
│   │                                                          │   │
│   │ 2. Routing                                              │   │
│   │    - Determine optimal path                             │   │
│   │    - Manage routing tables                              │   │
│   │                                                          │   │
│   │ 3. Packet Forwarding                                    │   │
│   │    - Forward packets to next hop                        │   │
│   │                                                          │   │
│   │ 4. Packet Fragmentation/Reassembly                      │   │
│   │    - Fragment packets according to MTU                  │   │
│   │    - Reassemble at destination                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Major Protocols:                                              │
│   ┌─────────────┬──────────────────────────────────────────┐    │
│   │    IP       │ Internet Protocol (IPv4, IPv6)           │    │
│   │    ICMP     │ Error reporting, ping                    │    │
│   │    ARP      │ IP → MAC address translation             │    │
│   │    RARP     │ MAC → IP address translation             │    │
│   │    OSPF     │ Routing protocol                         │    │
│   │    BGP      │ Inter-AS routing                         │    │
│   └─────────────┴──────────────────────────────────────────┘    │
│                                                                  │
│   Key Devices: Router, L3 Switch                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 2: Data Link Layer

```
┌─────────────────────────────────────────────────────────────────┐
│               Layer 2: Data Link                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Reliable frame transmission between adjacent nodes      │
│         "Error corrector of physical layer"                     │
│                                                                  │
│   Two Sublayers:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │              LLC (Logical Link Control)          │   │   │
│   │   │              Logical link control                 │   │   │
│   │   │   - Interface with upper layers                  │   │   │
│   │   │   - Flow control, error control                  │   │   │
│   │   │   - Frame synchronization                        │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │              MAC (Media Access Control)          │   │   │
│   │   │              Media access control                 │   │   │
│   │   │   - MAC addressing                               │   │   │
│   │   │   - Determine media access method                │   │   │
│   │   │   - Collision detection/avoidance                │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Functions:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. Framing                                              │   │
│   │    - Organize bit stream into frames                    │   │
│   │    - Add start/end delimiters                           │   │
│   │                                                          │   │
│   │ 2. Physical Addressing                                  │   │
│   │    - MAC address (48 bits)                              │   │
│   │    - Unique hardware address                            │   │
│   │                                                          │   │
│   │ 3. Error Detection                                      │   │
│   │    - CRC (Cyclic Redundancy Check)                      │   │
│   │    - Checksum                                           │   │
│   │                                                          │   │
│   │ 4. Media Access Control                                 │   │
│   │    - CSMA/CD (Ethernet)                                 │   │
│   │    - CSMA/CA (Wireless)                                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Frame Structure (Ethernet):                                   │
│   ┌──────┬───────┬───────┬──────┬─────────┬─────┐              │
│   │Pream │ Dest  │ Src   │ Type │  Data   │ FCS │              │
│   │ble   │ MAC   │ MAC   │      │         │     │              │
│   │(8B)  │(6B)   │(6B)   │(2B)  │(46-1500)│(4B) │              │
│   └──────┴───────┴───────┴──────┴─────────┴─────┘              │
│                                                                  │
│   Protocols/Technologies: Ethernet (802.3), Wi-Fi (802.11), PPP│
│   Key Devices: Switch, Bridge, NIC                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Physical Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                  Layer 1: Physical                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Convert bits (0 and 1) to physical signals and transmit │
│         "Foundation of the network"                             │
│                                                                  │
│   Bit Transmission:                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Digital Data: 1 0 1 1 0 0 1 0                         │   │
│   │                  │ │ │ │ │ │ │ │                        │   │
│   │                  ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼                        │   │
│   │                                                          │   │
│   │   Electrical:   ┌─┐ ┌─┐ ┌─┐   ┌─┐                       │   │
│   │   Signal       │ │ │ │ │ │   │ │                       │   │
│   │   (Wired)   ───┘ └─┘ └─┘ └───┘ └───                     │   │
│   │                                                          │   │
│   │   Optical:      ● ○ ● ● ○ ○ ● ○                         │   │
│   │   Signal        (Light pulses)                           │   │
│   │   (Fiber)                                                │   │
│   │                                                          │   │
│   │   Wireless:     ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿                        │   │
│   │   Signal        (Modulated electromagnetic wave)         │   │
│   │   (Wi-Fi)                                                │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Functions:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. Bit Transmission                                     │   │
│   │    - Convert 0 and 1 to signals                         │   │
│   │    - Restore signals to 0 and 1                         │   │
│   │                                                          │   │
│   │ 2. Define Physical Characteristics                      │   │
│   │    - Cable type, connector specs                        │   │
│   │    - Pin arrangement, voltage levels                    │   │
│   │                                                          │   │
│   │ 3. Define Transmission Mode                             │   │
│   │    - Simplex/duplex, synchronous/asynchronous           │   │
│   │                                                          │   │
│   │ 4. Define Data Transmission Rate                        │   │
│   │    - Bandwidth, bps (bits per second)                   │   │
│   │                                                          │   │
│   │ 5. Synchronization                                      │   │
│   │    - Timing alignment between sender and receiver       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Transmission Media:                                           │
│   ┌─────────────┬──────────────────────────────────────────┐    │
│   │   Wired     │ Coaxial, UTP, STP, Fiber optic           │    │
│   │   Wireless  │ Radio waves (Wi-Fi), Microwave, Infrared │    │
│   └─────────────┴──────────────────────────────────────────┘    │
│                                                                  │
│   Key Devices: Hub, Repeater, Cable, Connector, NIC             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Protocols by Layer

### Protocol Summary by Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                     Major Protocols by Layer                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer         Protocol                     Description         │
│  ─────────────────────────────────────────────────────────────│
│                                                                  │
│  7. Application HTTP, HTTPS     Web communication               │
│               FTP, SFTP        File transfer                    │
│               SMTP, POP3, IMAP Email                            │
│               DNS              Domain name resolution           │
│               DHCP             Automatic IP assignment          │
│               SSH, Telnet      Remote access                    │
│               SNMP             Network management               │
│               NTP              Time synchronization             │
│                                                                  │
│  6. Presentation SSL/TLS       Encryption                       │
│               JPEG, GIF, MPEG  Media formats                    │
│               ASCII, EBCDIC    Character encoding               │
│               XDR              Data representation              │
│                                                                  │
│  5. Session    NetBIOS         Network basic I/O                │
│               RPC              Remote procedure call            │
│               PPTP             Tunneling                        │
│               SIP              VoIP session control             │
│                                                                  │
│  4. Transport  TCP             Reliable transport               │
│               UDP              Unreliable fast transport        │
│               SCTP             Stream transport                 │
│               DCCP             Datagram congestion control      │
│                                                                  │
│  3. Network    IP (IPv4, IPv6) Internet protocol                │
│               ICMP             Error messages, ping             │
│               ARP, RARP        Address translation              │
│               OSPF, RIP, BGP   Routing protocols                │
│               IGMP             Multicast group management       │
│                                                                  │
│  2. Data Link  Ethernet (802.3) Wired LAN                       │
│               Wi-Fi (802.11)   Wireless LAN                     │
│               PPP              Point-to-point connection        │
│               HDLC             Data link control                │
│               Frame Relay      WAN protocol                     │
│                                                                  │
│  1. Physical   RS-232          Serial communication             │
│               RJ-45            Ethernet connector               │
│               IEEE 802.3       Ethernet physical spec           │
│               DSL              Digital subscriber line          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Major Port Numbers

```
┌─────────────────────────────────────────────────────────────────┐
│                      Major Port Numbers                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Well-Known Ports (0-1023)                                     │
│   ┌──────────┬──────────┬────────────────────────────────────┐  │
│   │ Port     │ Protocol │        Purpose                     │  │
│   ├──────────┼──────────┼────────────────────────────────────┤  │
│   │    20    │   FTP    │ Data transfer                      │  │
│   │    21    │   FTP    │ Control connection                 │  │
│   │    22    │   SSH    │ Secure shell                       │  │
│   │    23    │  Telnet  │ Remote access (insecure)           │  │
│   │    25    │   SMTP   │ Mail transmission                  │  │
│   │    53    │   DNS    │ Domain name resolution             │  │
│   │    67    │   DHCP   │ Server                             │  │
│   │    68    │   DHCP   │ Client                             │  │
│   │    80    │   HTTP   │ Web (unencrypted)                  │  │
│   │   110    │   POP3   │ Mail retrieval                     │  │
│   │   143    │   IMAP   │ Mail access                        │  │
│   │   443    │  HTTPS   │ Web (encrypted)                    │  │
│   │   3389   │   RDP    │ Remote desktop                     │  │
│   └──────────┴──────────┴────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. PDU (Protocol Data Unit)

### PDU Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                  PDU (Protocol Data Unit)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PDU: Unit of data handled at each layer                       │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Layer            PDU Name        Components           │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │  7. Application   Data             User data           │   │
│   │  6. Presentation  Data             Encoded data        │   │
│   │  5. Session       Data             Session data        │   │
│   │  4. Transport     Segment          Header + data       │   │
│   │                   or Datagram                          │   │
│   │  3. Network       Packet           Header + segment    │   │
│   │  2. Data Link     Frame            Header + packet +   │   │
│   │                                    trailer             │   │
│   │  1. Physical      Bit              Stream of 0s and 1s │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   PDU Structure by Layer:                                       │
│                                                                  │
│   Application/Presentation/Session:                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                        DATA                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Transport Layer (Segment):                                    │
│   ┌──────────────┬──────────────────────────────────────────┐   │
│   │ TCP/UDP Hdr  │                    DATA                   │   │
│   └──────────────┴──────────────────────────────────────────┘   │
│                                                                  │
│   Network Layer (Packet):                                       │
│   ┌───────────┬──────────────┬──────────────────────────────┐   │
│   │  IP Hdr   │ TCP/UDP Hdr  │            DATA               │   │
│   └───────────┴──────────────┴──────────────────────────────┘   │
│                                                                  │
│   Data Link Layer (Frame):                                      │
│   ┌────────┬───────────┬──────────────┬─────────────┬───────┐   │
│   │Preamble│Ether Hdr  │   IP Hdr    │   Segment   │  FCS  │   │
│   └────────┴───────────┴──────────────┴─────────────┴───────┘   │
│                                                                  │
│   Physical Layer (Bits):                                        │
│   10110010 11010101 00101101 11100010 ...                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### TCP vs UDP Segment

```
┌─────────────────────────────────────────────────────────────────┐
│                   TCP Segment Structure (20+ bytes)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    0                   1                   2                   3 │
│    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |          Source Port          |       Destination Port        │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |                        Sequence Number                        │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |                    Acknowledgment Number                      │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |  Data |           |U|A|P|R|S|F|                               │
│   | Offset| Reserved  |R|C|S|S|Y|I|            Window             │
│   |       |           |G|K|H|T|N|N|                               │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |           Checksum            |         Urgent Pointer        │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |                    Options (if any)                           │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |                             Data                              │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│                                                                  │
│   Flags: SYN(connect), ACK(ack), FIN(terminate),                │
│          RST(force term), PSH(push), URG(urgent)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   UDP Datagram Structure (8 bytes)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    0                   1                   2                   3 │
│    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |          Source Port          |       Destination Port        │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |            Length             |           Checksum            │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│   |                             Data                              │
│   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+│
│                                                                  │
│   Much simpler than TCP → Less overhead → Faster transmission   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Encapsulation and Decapsulation

### Encapsulation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Encapsulation                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Process where headers (control information) of each layer     │
│   are added as data travels from upper to lower layers          │
│                                                                  │
│   Sender Side (Data Transmission Process):                      │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Application Layer                                       │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │                   DATA                           │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Presentation Layer                                      │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │               DATA (encrypted/compressed)        │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Session Layer                                           │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │                   DATA                           │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Transport Layer                                         │   │
│   │  ┌──────────┬──────────────────────────────────────┐    │   │
│   │  │ TCP Hdr  │              DATA                     │    │   │
│   │  └──────────┴──────────────────────────────────────┘    │   │
│   │       ↑            Segment                              │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Network Layer                                           │   │
│   │  ┌────────┬──────────┬──────────────────────────────┐   │   │
│   │  │IP Hdr  │ TCP Hdr  │            DATA               │   │   │
│   │  └────────┴──────────┴──────────────────────────────┘   │   │
│   │      ↑            Packet                                │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Data Link Layer                                         │   │
│   │  ┌──────┬────────┬──────────┬───────────────────┬─────┐│   │
│   │  │ Hdr  │IP Hdr  │ TCP Hdr  │       DATA        │ FCS ││   │
│   │  └──────┴────────┴──────────┴───────────────────┴─────┘│   │
│   │     ↑            Frame                            ↑    │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Physical Layer                                          │   │
│   │  10110100 01101011 11010010 10101100 ...                │   │
│   │                    Bits                                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Decapsulation

```
┌─────────────────────────────────────────────────────────────────┐
│                  Decapsulation                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Process where headers of each layer are removed as data       │
│   travels from lower to upper layers at the receiver            │
│                                                                  │
│   Receiver Side (Data Reception Process):                       │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Physical Layer                                          │   │
│   │  10110100 01101011 11010010 10101100 ...                │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Data Link Layer                                         │   │
│   │  ┌──────┬────────┬──────────┬───────────────────┬─────┐│   │
│   │  │ Hdr  │IP Hdr  │ TCP Hdr  │       DATA        │ FCS ││   │
│   │  └──────┴────────┴──────────┴───────────────────┴─────┘│   │
│   │     │    │    Verify FCS, remove header/trailer      │  │   │
│   │     ▼    │                                              │   │
│   │  [Remove]│                                              │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Network Layer                                           │   │
│   │  ┌────────┬──────────┬──────────────────────────────┐   │   │
│   │  │IP Hdr  │ TCP Hdr  │            DATA               │   │   │
│   │  └────────┴──────────┴──────────────────────────────┘   │   │
│   │     │        Verify IP header, remove                   │   │
│   │     ▼                                                    │   │
│   │  [Remove]                                                │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Transport Layer                                         │   │
│   │  ┌──────────┬──────────────────────────────────────┐    │   │
│   │  │ TCP Hdr  │              DATA                     │    │   │
│   │  └──────────┴──────────────────────────────────────┘    │   │
│   │      │         Verify TCP header, remove                │   │
│   │      ▼                                                   │   │
│   │   [Remove]                                               │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │  Session/Presentation/Application Layer                  │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │                   DATA                           │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   │                                                          │   │
│   │                   Delivered to application               │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Complete Encapsulation/Decapsulation Process

```
┌─────────────────────────────────────────────────────────────────┐
│                Complete Communication (Send → Receive)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│      Sending Host                        Receiving Host         │
│   ┌─────────────────┐               ┌─────────────────┐         │
│   │  Application    │               │  Application    │         │
│   │  [Generate DATA]│               │  [Use DATA]     │         │
│   └────────┬────────┘               └────────▲────────┘         │
│            │ Encapsulation                   │ Decapsulation    │
│   ┌────────▼────────┐               ┌────────┴────────┐         │
│   │  Presentation   │               │  Presentation   │         │
│   │  [Encrypt/      │               │  [Decrypt/      │         │
│   │   Compress]     │               │   Decompress]   │         │
│   └────────┬────────┘               └────────▲────────┘         │
│            │                                 │                   │
│   ┌────────▼────────┐               ┌────────┴────────┐         │
│   │    Session      │               │    Session      │         │
│   │  [Manage        │               │  [Manage        │         │
│   │   Session]      │               │   Session]      │         │
│   └────────┬────────┘               └────────▲────────┘         │
│            │                                 │                   │
│   ┌────────▼────────┐               ┌────────┴────────┐         │
│   │   Transport     │               │   Transport     │         │
│   │  [+TCP Header]  │               │  [-TCP Header]  │         │
│   └────────┬────────┘               └────────▲────────┘         │
│            │ Segment                         │                   │
│   ┌────────▼────────┐               ┌────────┴────────┐         │
│   │    Network      │               │    Network      │         │
│   │  [+IP Header]   │               │  [-IP Header]   │         │
│   └────────┬────────┘               └────────▲────────┘         │
│            │ Packet                          │                   │
│   ┌────────▼────────┐               ┌────────┴────────┐         │
│   │   Data Link     │               │   Data Link     │         │
│   │ [+Eth Hdr+FCS]  │               │ [-Eth Hdr-FCS]  │         │
│   └────────┬────────┘               └────────▲────────┘         │
│            │ Frame                           │                   │
│   ┌────────▼────────┐               ┌────────┴────────┐         │
│   │    Physical     │               │    Physical     │         │
│   │  [Transmit Bits]│──────────────►│  [Receive Bits] │         │
│   └─────────────────┘   Network     └─────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Key Devices by Layer

### Device Mapping by Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                     Network Devices by Layer                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer                    Device                                │
│  ─────────────────────────────────────────────────────────────│
│                                                                  │
│  7. Application  ┌────────────────────────────────────────────┐ │
│                  │ Firewall (L7/Application), Proxy server,   │ │
│                  │ Load balancer, ADC, IDS/IPS                │ │
│                  └────────────────────────────────────────────┘ │
│                                                                  │
│  6. Presentation ┌────────────────────────────────────────────┐ │
│                  │ Software (SSL/TLS library)                 │ │
│                  └────────────────────────────────────────────┘ │
│                                                                  │
│  5. Session      ┌────────────────────────────────────────────┐ │
│                  │ Software (session manager)                 │ │
│                  └────────────────────────────────────────────┘ │
│                                                                  │
│  4. Transport    ┌────────────────────────────────────────────┐ │
│                  │ Firewall (L4), Load balancer              │ │
│                  └────────────────────────────────────────────┘ │
│                                                                  │
│  3. Network      ┌────────────────────────────────────────────┐ │
│                  │ Router, L3 Switch, Firewall (L3)           │ │
│                  └────────────────────────────────────────────┘ │
│                                                                  │
│  2. Data Link    ┌────────────────────────────────────────────┐ │
│                  │ Switch (L2), Bridge, NIC, Wireless AP      │ │
│                  └────────────────────────────────────────────┘ │
│                                                                  │
│  1. Physical     ┌────────────────────────────────────────────┐ │
│                  │ Hub, Repeater, Cable, Connector, Modem     │ │
│                  └────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Device Operating Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   Device Operating Layers                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│           Hub        Switch       Router      Firewall(L7)      │
│                                                                  │
│   7. Application │          │           │           ████████    │
│   6. Presentation│          │           │           ████████    │
│   5. Session   │          │           │           ████████      │
│   4. Transport │          │           │           ████████      │
│   3. Network   │          │           ████████   ████████      │
│   2. Data Link │        ████████   ████████   ████████         │
│   1. Physical  ████████   ████████   ████████   ████████        │
│                                                                  │
│   Hub: L1 only (signal amplification/replication)               │
│   Switch: L1-L2 (MAC-based forwarding)                          │
│   Router: L1-L3 (IP-based routing)                              │
│   Firewall(L7): L1-L7 (deep packet inspection)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Practical Application of OSI Model

### Web Page Loading Process

```
┌─────────────────────────────────────────────────────────────────┐
│          Web Page Loading (Accessing www.example.com)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User enters www.example.com in browser                        │
│                                                                  │
│   7. Application Layer                                          │
│      └── Generate HTTP request: GET / HTTP/1.1                  │
│          Host: www.example.com                                  │
│                                                                  │
│   6. Presentation Layer                                         │
│      └── Apply TLS encryption if HTTPS                          │
│          Data encoding (UTF-8)                                  │
│                                                                  │
│   5. Session Layer                                              │
│      └── Manage TCP connection session                          │
│          Handle cookies/session ID                              │
│                                                                  │
│   4. Transport Layer                                            │
│      └── Create TCP segment                                     │
│          Ports: Source(49152), Dest(443)                        │
│          Assign sequence number                                 │
│                                                                  │
│   3. Network Layer                                              │
│      └── Create IP packet                                       │
│          Resolve domain → IP via DNS                            │
│          Source IP: 192.168.1.100                               │
│          Dest IP: 93.184.216.34                                 │
│                                                                  │
│   2. Data Link Layer                                            │
│      └── Create Ethernet frame                                  │
│          Add MAC address (verify via ARP)                       │
│          Add CRC checksum                                       │
│                                                                  │
│   1. Physical Layer                                             │
│      └── Convert to electrical signals and transmit via cable   │
│          (or transmit as wireless signal)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Network Troubleshooting

```
┌─────────────────────────────────────────────────────────────────┐
│                OSI Model-based Troubleshooting                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Check sequentially from lower to upper layers                 │
│                                                                  │
│   1. Physical Layer Check                                       │
│      ┌────────────────────────────────────────────────────┐     │
│      │ - Check cable connection status                    │     │
│      │ - Check LED indicators                             │     │
│      │ - Check NIC status                                 │     │
│      │ - ping localhost (127.0.0.1)                       │     │
│      └────────────────────────────────────────────────────┘     │
│                                                                  │
│   2. Data Link Layer Check                                      │
│      ┌────────────────────────────────────────────────────┐     │
│      │ - Check MAC address (ipconfig /all, ifconfig)      │     │
│      │ - Check ARP table (arp -a)                         │     │
│      │ - Check switch MAC table                           │     │
│      │ - Check duplex settings                            │     │
│      └────────────────────────────────────────────────────┘     │
│                                                                  │
│   3. Network Layer Check                                        │
│      ┌────────────────────────────────────────────────────┐     │
│      │ - Check IP configuration (ipconfig, ifconfig)      │     │
│      │ - Ping default gateway                             │     │
│      │ - Check routing table (netstat -r, route print)    │     │
│      │ - Trace route with traceroute                      │     │
│      └────────────────────────────────────────────────────┘     │
│                                                                  │
│   4. Transport Layer Check                                      │
│      ┌────────────────────────────────────────────────────┐     │
│      │ - Check port status (netstat -an)                  │     │
│      │ - Check firewall rules                             │     │
│      │ - Test port connection with telnet                 │     │
│      └────────────────────────────────────────────────────┘     │
│                                                                  │
│   5-7. Upper Layer Check                                        │
│      ┌────────────────────────────────────────────────────┐     │
│      │ - Check application logs                           │     │
│      │ - Check DNS resolution (nslookup)                  │     │
│      │ - Check service status                             │     │
│      │ - Verify certificate validity (HTTPS)              │     │
│      └────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Troubleshooting Command Summary

> **Why bottom-up?** Lower layers affect everything above them — a broken cable (L1) makes DNS debugging (L7) pointless. By starting at L1 and working up, you avoid chasing symptoms caused by a lower-layer failure. At each layer, the tool is chosen to test *that layer's responsibility specifically*: `arp -a` tests L2 address resolution (not IP routing), `ping` tests L3 reachability (not port connectivity), and `netstat -an` tests L4 port status (not application logic).

```
┌─────────────────────────────────────────────────────────────────┐
│                   Troubleshooting Commands                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Layer    Command (Windows/Linux)     Purpose                  │
│   ─────────────────────────────────────────────────────────────│
│                                                                  │
│   L1      - Check LED                  Physical connection state│
│           - Cable tester                                        │
│                                                                  │
│   L2      - arp -a                     Check ARP table          │
│           - ipconfig /all | ifconfig   Check MAC address        │
│                                                                  │
│   L3      - ping <IP>                  Connection test          │
│           - tracert | traceroute       Path trace               │
│           - ipconfig | ifconfig        Check IP configuration   │
│           - netstat -r | route         Routing table            │
│                                                                  │
│   L4      - netstat -an                Port status              │
│           - telnet <IP> <port>         Port connection test     │
│                                                                  │
│   L5-7    - nslookup <domain>          DNS check                │
│           - curl | wget                HTTP test                │
│           - Application logs                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Basic Problems

**1. List the OSI 7 layers in order from bottom to top.**

**2. Match the following protocols to their corresponding layers:**
   - (a) HTTP       (  ) Transport Layer
   - (b) TCP        (  ) Network Layer
   - (c) IP         (  ) Data Link Layer
   - (d) Ethernet   (  ) Application Layer

**3. Choose the correct PDU matching:**
   - Transport Layer: (  )
   - Network Layer: (  )
   - Data Link Layer: (  )

   Options: Segment, Packet, Frame, Bit

**4. Explain the order in which headers are added during encapsulation.**

### Applied Problems

**5. Estimate the layer where the problem occurred in the following situations:**
   - (a) Connected cable but LED doesn't light up
   - (b) Can ping other PCs on same network but no internet
   - (c) Can access web page but cannot log in

**6. Explain the role of each OSI layer in HTTP and HTTPS communication.**

**7. Explain the differences between TCP and UDP from the OSI model perspective.**

### Advanced Problems

**8. Indicate which OSI layers each device processes in the diagram below:**

```
[PC] ---[Hub]---[Switch]---[Router]---[Firewall]---[Server]
```

**9. Explain why encapsulation and decapsulation are necessary.**

**10. Explain the differences between the OSI model and TCP/IP model from a layer structure perspective.**

---

<details>
<summary>Answers</summary>

**1.** Physical → Data Link → Network → Transport → Session → Presentation → Application

**2.**
- (a) HTTP - Application Layer
- (b) TCP - Transport Layer
- (c) IP - Network Layer
- (d) Ethernet - Data Link Layer

**3.**
- Transport Layer: Segment
- Network Layer: Packet
- Data Link Layer: Frame

**4.** Application data → Add TCP header (segment) → Add IP header (packet) → Add Ethernet header/trailer (frame) → Convert to bits

**5.**
- (a) Physical Layer (cable or NIC problem)
- (b) Network Layer (routing/gateway problem)
- (c) Application Layer or Session Layer (authentication/session problem)

**6.**
- Application: Generate/process HTTP requests/responses
- Presentation: TLS encryption for HTTPS
- Session: TCP connection management
- Transport: TCP segmentation, port numbers
- Network: IP packets, routing
- Data Link: MAC addresses, frames
- Physical: Bit transmission

**7.**
- TCP: Connection-oriented, reliability guaranteed, flow/error control, segment-based
- UDP: Connectionless, unreliable, low overhead, datagram-based
- Both operate at Transport Layer (L4)

**8.**
- Hub: L1
- Switch: L2
- Router: L3
- Firewall: L3-L7 (depends on type)

**9.**
- Maintain layer independence (changes in one layer don't affect others)
- Provide standardized interfaces
- Ensure interoperability
- Modular approach facilitates development/maintenance

**10.**
- OSI: 7 layers, theoretical reference model, ISO standard
- TCP/IP: 4 layers, practical implementation model, Internet standard
- OSI's Session/Presentation layers integrated into Application layer in TCP/IP

</details>

---

**Previous**: [Network Fundamentals](./01_Network_Fundamentals.md) | **Next**: [TCP/IP Model](./03_TCP_IP_Model.md)

---

## References

- Computer Networking: A Top-Down Approach (Kurose & Ross)
- TCP/IP Illustrated (W. Richard Stevens)
- [Cisco: OSI Model](https://www.cisco.com/c/en/us/solutions/small-business/resource-center/networking/osi-model.html)
- [RFC 1122: Requirements for Internet Hosts](https://tools.ietf.org/html/rfc1122)
