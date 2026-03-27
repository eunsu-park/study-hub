# Data Link Layer

**Previous**: [Physical Layer](./04_Physical_Layer.md) | **Next**: [IP Addressing and Subnetting](./06_IP_Address_Subnetting.md)

**Difficulty**: ⭐⭐ (Beginner-Intermediate)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of the Data Link Layer and distinguish between its LLC and MAC sublayers
2. Describe the structure of a MAC address and interpret OUI and special addresses (broadcast, multicast)
3. Analyze the Ethernet II frame structure and identify each field's purpose
4. Explain how CSMA/CD works and why modern full-duplex Ethernet does not require it
5. Trace the switch learning, forwarding, flooding, and filtering process
6. Describe how ARP resolves IP addresses to MAC addresses
7. Distinguish between collision domains and broadcast domains and explain how hubs, switches, and routers affect each

---

The Data Link Layer bridges the gap between raw physical signals and meaningful network communication. It is the layer that gives your computer a hardware identity (MAC address), packages bits into frames, detects transmission errors, and ensures that data reaches the correct neighbor on a shared medium. Understanding this layer is essential for troubleshooting LAN issues, configuring switches, and grasping how Ethernet -- the dominant LAN technology -- actually works.

## Table of Contents

1. [Role of the Data Link Layer](#1-role-of-the-data-link-layer)
2. [MAC Address](#2-mac-address)
3. [Frame Structure](#3-frame-structure)
4. [Ethernet (IEEE 802.3)](#4-ethernet-ieee-8023)
5. [Switch Operation Principles](#5-switch-operation-principles)
6. [ARP (Address Resolution Protocol)](#6-arp-address-resolution-protocol)
7. [Collision Domain and Broadcast Domain](#7-collision-domain-and-broadcast-domain)
8. [Practice Problems](#8-practice-problems)

---

## 1. Role of the Data Link Layer

### Definition of the Data Link Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                 Data Link Layer (Layer 2)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   "Layer responsible for reliable data transmission between     │
│    adjacent nodes"                                               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Network Layer (IP Packet)                             │   │
│   │         │                                                │   │
│   │         ▼                                                │   │
│   │   ┌─────────────────────────────────────┐               │   │
│   │   │       Data Link Layer                │               │   │
│   │   │                                       │               │   │
│   │   │   - Framing                           │               │   │
│   │   │   - MAC Addressing                    │               │   │
│   │   │   - Error Detection                   │               │   │
│   │   │   - Media Access Control              │               │   │
│   │   │                                       │               │   │
│   │   └───────────────┬─────────────────────┘               │   │
│   │                   │                                      │   │
│   │                   ▼                                      │   │
│   │   Physical Layer (Bit Stream)                           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Node-to-Node vs End-to-End:                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Host A     Router1     Router2     Host B             │   │
│   │   ┌───┐      ┌───┐      ┌───┐      ┌───┐              │   │
│   │   │   │◄────►│   │◄────►│   │◄────►│   │              │   │
│   │   └───┘      └───┘      └───┘      └───┘              │   │
│   │       ◄─L2──► ◄─L2──► ◄─L2──►                         │   │
│   │     Node-to- Node-to- Node-to-                         │   │
│   │      Node     Node     Node                            │   │
│   │                                                          │   │
│   │       ◄─────────────── L3 (IP) ───────────────►         │   │
│   │                End-to-End                                │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Sub-layers of the Data Link Layer

```
┌─────────────────────────────────────────────────────────────────┐
│              Two Sub-layers of the Data Link Layer               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   The Data Link Layer consists of two sub-layers:               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │           LLC (Logical Link Control)             │   │   │
│   │   │             Logical Link Control                 │   │   │
│   │   │                                                   │   │   │
│   │   │   - Interface with upper layer (Network)         │   │   │
│   │   │   - Flow control, error control                  │   │   │
│   │   │   - Multiplexing (support multiple protocols)    │   │   │
│   │   │   - IEEE 802.2                                   │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │           MAC (Media Access Control)             │   │   │
│   │   │             Media Access Control                 │   │   │
│   │   │                                                   │   │   │
│   │   │   - Physical addressing (MAC address)            │   │   │
│   │   │   - Framing (frame boundary definition)          │   │   │
│   │   │   - Media access method (CSMA/CD, CSMA/CA)       │   │   │
│   │   │   - Error detection (CRC)                        │   │   │
│   │   │   - IEEE 802.3 (Ethernet), 802.11 (Wi-Fi)        │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Most modern Ethernet does not use LLC; instead,               │
│   it identifies upper protocols using the EtherType field.      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Key Functions

```
┌─────────────────────────────────────────────────────────────────┐
│                 Key Functions of the Data Link Layer             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Framing                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Organize bit stream into meaningful frame units       │   │
│   │                                                          │   │
│   │   Bit Stream: 101101001011010010110100...              │   │
│   │                  ↓                                       │   │
│   │   Frame:    [Header|  Data  |Trailer]                   │   │
│   │                                                          │   │
│   │   Frame delimitation methods:                           │   │
│   │   - Length-based: Specify frame size                    │   │
│   │   - Flag-based: Start/end markers (e.g., HDLC 01111110) │   │
│   │   - Preamble: Ethernet start pattern                    │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. Physical Addressing                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Identify devices within the same network using MAC    │   │
│   │   addresses                                              │   │
│   │                                                          │   │
│   │   Example: 00:1A:2B:3C:4D:5E                            │   │
│   │       ↑                 ↑                               │   │
│   │       OUI (Vendor)      Unique number                   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. Error Detection                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Detect errors during transmission (limited correction)│   │
│   │                                                          │   │
│   │   Methods:                                               │   │
│   │   - CRC (Cyclic Redundancy Check): Most common          │   │
│   │   - Parity bit: Simple but limited                      │   │
│   │   - Checksum: Sum-based                                 │   │
│   │                                                          │   │
│   │   When error found: Discard frame (retransmit at upper) │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   4. Media Access Control                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Determine when to transmit on shared media            │   │
│   │                                                          │   │
│   │   Methods:                                               │   │
│   │   - CSMA/CD: Ethernet (collision detection)             │   │
│   │   - CSMA/CA: Wireless (collision avoidance)             │   │
│   │   - Token passing: Token Ring                           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   5. Flow Control                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Adjust transmission rate to receiver's capacity       │   │
│   │                                                          │   │
│   │   - Supported only in some data link protocols          │   │
│   │   - Ethernet: PAUSE frame (802.3x)                      │   │
│   │   - Most rely on TCP/IP flow control                    │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. MAC Address

### MAC Address Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                 MAC Address (Media Access Control Address)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Definition: Unique physical address assigned to network       │
│              interface, also called "hardware address" or       │
│              "Ethernet address"                                  │
│                                                                  │
│   Structure (48 bits = 6 bytes):                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   00:1A:2B:3C:4D:5E                                     │   │
│   │   ├─────┤ ├─────────┤                                   │   │
│   │     OUI      NIC Unique ID                               │   │
│   │   (24 bits)    (24 bits)                                 │   │
│   │                                                          │   │
│   │   OUI (Organizationally Unique Identifier):             │   │
│   │   - Manufacturer identification code (assigned by IEEE) │   │
│   │   - Example: 00:1A:2B = specific manufacturer           │   │
│   │                                                          │   │
│   │   NIC Unique ID:                                        │   │
│   │   - Assigned by manufacturer to each device             │   │
│   │   - Must be unique within same OUI                      │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Notation formats:                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   - Colon delimiter: 00:1A:2B:3C:4D:5E  (Unix/Linux)    │   │
│   │   - Dash delimiter: 00-1A-2B-3C-4D-5E  (Windows)        │   │
│   │   - Dot delimiter:   001A.2B3C.4D5E     (Cisco)         │   │
│   │   - Continuous:      001A2B3C4D5E                        │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Byte structure details:                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   First byte bit structure:                             │   │
│   │   ┌─┬─┬─┬─┬─┬─┬─┬─┐                                    │   │
│   │   │7│6│5│4│3│2│1│0│                                    │   │
│   │   └─┴─┴─┴─┴─┴─┴─┴─┘                                    │   │
│   │               │ │                                       │   │
│   │               │ └── I/G (Individual/Group)              │   │
│   │               │      0: Unicast (individual address)    │   │
│   │               │      1: Multicast (group address)       │   │
│   │               │                                         │   │
│   │               └──── U/L (Universal/Local)               │   │
│   │                      0: Globally unique (IEEE assigned) │   │
│   │                      1: Locally administered (user set) │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Special MAC Addresses

```
┌─────────────────────────────────────────────────────────────────┐
│                      Special MAC Addresses                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Broadcast Address                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   FF:FF:FF:FF:FF:FF                                     │   │
│   │                                                          │   │
│   │   - All bits set to 1                                   │   │
│   │   - Sent to all devices on the same network             │   │
│   │   - Used in ARP requests                                │   │
│   │                                                          │   │
│   │   Sender ─────────► All hosts                           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. Multicast Address                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   First byte LSB = 1 (odd number)                       │   │
│   │                                                          │   │
│   │   IPv4 Multicast: 01:00:5E:xx:xx:xx                     │   │
│   │   IPv6 Multicast: 33:33:xx:xx:xx:xx                     │   │
│   │   STP:            01:80:C2:00:00:00                      │   │
│   │                                                          │   │
│   │   Sender ─────────► Specific group of hosts only        │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. Unicast Address                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   First byte LSB = 0 (even number)                      │   │
│   │                                                          │   │
│   │   Example: 00:1A:2B:3C:4D:5E                            │   │
│   │                                                          │   │
│   │   Sender ─────────► Specific single host                │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Common OUI Examples:                                          │
│   ┌──────────────────┬───────────────────────────────────────┐  │
│   │       OUI        │           Manufacturer                 │  │
│   ├──────────────────┼───────────────────────────────────────┤  │
│   │  00:00:0C        │  Cisco                                │  │
│   │  00:0C:29        │  VMware                               │  │
│   │  00:50:56        │  VMware                               │  │
│   │  00:1A:A0        │  Dell                                 │  │
│   │  00:25:00        │  Apple                                │  │
│   │  AC:DE:48        │  Intel                                │  │
│   │  F0:1F:AF        │  Hewlett Packard                      │  │
│   └──────────────────┴───────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### MAC Address Lookup Commands

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAC Address Lookup Commands                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Windows:                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   C:\> ipconfig /all                                     │   │
│   │                                                          │   │
│   │   Ethernet adapter Ethernet0:                            │   │
│   │      Physical Address. . . . . . . : 00-1A-2B-3C-4D-5E  │   │
│   │      IPv4 Address. . . . . . . . . : 192.168.1.100      │   │
│   │                                                          │   │
│   │   C:\> getmac                                            │   │
│   │   Physical Address    Transport Name                     │   │
│   │   =================== =================================  │   │
│   │   00-1A-2B-3C-4D-5E   \Device\Tcpip_{GUID}              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Linux/Mac:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   $ ifconfig                                             │   │
│   │   eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>      │   │
│   │         ether 00:1a:2b:3c:4d:5e  txqueuelen 1000        │   │
│   │         inet 192.168.1.100  netmask 255.255.255.0       │   │
│   │                                                          │   │
│   │   $ ip link show                                         │   │
│   │   2: eth0: <BROADCAST,MULTICAST,UP>                      │   │
│   │       link/ether 00:1a:2b:3c:4d:5e brd ff:ff:ff:ff:ff:ff│   │
│   │                                                          │   │
│   │   Mac:                                                   │   │
│   │   $ networksetup -listallhardwareports                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Frame Structure

### Ethernet Frame Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ethernet Frame Structure                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Ethernet II (DIX) Frame (most common):                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │  ┌───────┬───────┬───────┬───────┬───────┬───────────┐  │   │
│   │  │ Pream │  SFD  │ Dest  │  Src  │EtherType│  Data   │  │   │
│   │  │  ble  │       │  MAC  │  MAC  │  /Len  │ Payload │  │   │
│   │  │  7B   │  1B   │  6B   │  6B   │   2B   │46-1500B │  │   │
│   │  └───────┴───────┴───────┴───────┴───────┴───────────┘  │   │
│   │                                                          │   │
│   │  ┌───────┐                                               │   │
│   │  │  FCS  │   ← Frame Check Sequence (CRC-32)            │   │
│   │  │  4B   │                                               │   │
│   │  └───────┘                                               │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Field Descriptions:                                           │
│   ┌───────────────┬──────────────────────────────────────────┐  │
│   │     Field     │              Description                  │  │
│   ├───────────────┼──────────────────────────────────────────┤  │
│   │ Preamble      │ Sync pattern (10101010... 7 bytes)       │  │
│   │ (7 bytes)     │                                          │  │
│   ├───────────────┼──────────────────────────────────────────┤  │
│   │ SFD           │ Frame start delimiter (10101011)         │  │
│   │ (1 byte)      │ Start Frame Delimiter                    │  │
│   ├───────────────┼──────────────────────────────────────────┤  │
│   │ Dest MAC      │ Destination MAC address                  │  │
│   │ (6 bytes)     │                                          │  │
│   ├───────────────┼──────────────────────────────────────────┤  │
│   │ Src MAC       │ Source MAC address                       │  │
│   │ (6 bytes)     │                                          │  │
│   ├───────────────┼──────────────────────────────────────────┤  │
│   │ EtherType     │ Upper layer protocol identifier          │  │
│   │ (2 bytes)     │ 0x0800=IPv4, 0x0806=ARP, 0x86DD=IPv6     │  │
│   ├───────────────┼──────────────────────────────────────────┤  │
│   │ Payload       │ Upper layer data (IP packet, etc.)       │  │
│   │ (46-1500 B)   │ Min 46, Max 1500 (Jumbo: 9000)          │  │
│   ├───────────────┼──────────────────────────────────────────┤  │
│   │ FCS           │ CRC-32 checksum for error detection      │  │
│   │ (4 bytes)     │                                          │  │
│   └───────────────┴──────────────────────────────────────────┘  │
│                                                                  │
│   Frame size:                                                   │
│   - Minimum: 64 bytes (Header 14 + Payload 46 + FCS 4)         │
│   - Maximum: 1518 bytes (Header 14 + Payload 1500 + FCS 4)     │
│   - Jumbo frame: Up to 9000 bytes payload                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### EtherType Values

```
┌─────────────────────────────────────────────────────────────────┐
│                     Common EtherType Values                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┬────────────────────────────────────────────┐ │
│   │  EtherType   │                Protocol                     │ │
│   ├──────────────┼────────────────────────────────────────────┤ │
│   │   0x0800     │  IPv4 (Internet Protocol version 4)        │ │
│   │   0x0806     │  ARP (Address Resolution Protocol)         │ │
│   │   0x8035     │  RARP (Reverse ARP)                        │ │
│   │   0x8100     │  VLAN (802.1Q)                             │ │
│   │   0x86DD     │  IPv6 (Internet Protocol version 6)        │ │
│   │   0x8847     │  MPLS (Unicast)                            │ │
│   │   0x8848     │  MPLS (Multicast)                          │ │
│   │   0x8863     │  PPPoE Discovery                           │ │
│   │   0x8864     │  PPPoE Session                             │ │
│   │   0x88CC     │  LLDP (Link Layer Discovery Protocol)      │ │
│   └──────────────┴────────────────────────────────────────────┘ │
│                                                                  │
│   * If EtherType < 0x0600, interpreted as length field (802.3)  │
│   * If EtherType ≥ 0x0600, interpreted as EtherType (Ethernet II)│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CRC-32 Error Detection

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRC-32 Error Detection                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   CRC (Cyclic Redundancy Check):                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Error detection code using polynomial division        │   │
│   │                                                          │   │
│   │   Transmission:                                          │   │
│   │   ┌────────────────────────────────────┐                │   │
│   │   │ Data (Header + Payload)             │                │   │
│   │   └──────────────────┬─────────────────┘                │   │
│   │                      │                                   │   │
│   │                      ▼                                   │   │
│   │              ┌───────────────┐                          │   │
│   │              │ CRC-32 Calc   │                          │   │
│   │              └───────┬───────┘                          │   │
│   │                      │                                   │   │
│   │                      ▼                                   │   │
│   │   ┌────────────────────────────────────┬──────┐         │   │
│   │   │ Data (Header + Payload)             │ FCS  │         │   │
│   │   └────────────────────────────────────┴──────┘         │   │
│   │                                                          │   │
│   │   Reception:                                             │   │
│   │   ┌────────────────────────────────────┬──────┐         │   │
│   │   │ Received Data                       │ FCS  │         │   │
│   │   └──────────────────┬─────────────────┴──────┘         │   │
│   │                      │                                   │   │
│   │                      ▼                                   │   │
│   │              ┌───────────────┐                          │   │
│   │              │ CRC-32 Recalc │                          │   │
│   │              └───────┬───────┘                          │   │
│   │                      │                                   │   │
│   │          Result = 0  │    Result ≠ 0                    │   │
│   │          ↓           │           ↓                      │   │
│   │      No error    Discard frame                          │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   CRC-32 Polynomial: x^32 + x^26 + x^23 + ... + 1              │
│                                                                  │
│   Detectable errors:                                            │
│   - 1-bit errors: 100% detection                               │
│   - 2-bit errors: 100% detection                               │
│   - Odd number of bit errors: 100% detection                   │
│   - Burst errors (≤32 bits): 100% detection                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Ethernet (IEEE 802.3)

### History and Evolution of Ethernet

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ethernet Evolution History                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Year    Name            Speed        Media                    │
│   ─────────────────────────────────────────────────────────────│
│   1973    Experimental    2.94 Mbps   Coax (Xerox PARC)         │
│   1983    10BASE5         10 Mbps     Thick Coax               │
│   1985    10BASE2         10 Mbps     Thin Coax                │
│   1990    10BASE-T        10 Mbps     UTP (Cat3)               │
│   1995    100BASE-TX      100 Mbps    UTP (Cat5)               │
│   1998    1000BASE-T      1 Gbps      UTP (Cat5e)              │
│   2006    10GBASE-T       10 Gbps     UTP (Cat6/6a)            │
│   2016    25GBASE-T       25 Gbps     UTP (Cat8)               │
│   2017    40GBASE-T       40 Gbps     UTP (Cat8)               │
│                                                                  │
│   Ethernet naming convention:                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   10    BASE   -   T                                    │   │
│   │   ↓      ↓         ↓                                    │   │
│   │   Speed  Signal    Media/Encoding                       │   │
│   │   (Mbps) (Baseband)                                     │   │
│   │                                                          │   │
│   │   Speed: 10, 100, 1000, 10G, 25G, 40G, 100G            │   │
│   │   Signal: BASE (Baseband), BROAD (Broadband-legacy)    │   │
│   │   Media: T (Twisted Pair), F/X (Fiber), S (Short wave) │   │
│   │         L (Long wave), C (Copper)                       │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CSMA/CD (Carrier Sense Multiple Access with Collision Detection)

```
┌─────────────────────────────────────────────────────────────────┐
│                         CSMA/CD                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   CSMA/CD: Media access method for half-duplex Ethernet         │
│                                                                  │
│   Operation process:                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   1. Carrier Sense                                      │   │
│   │      ┌──────────┐                                       │   │
│   │      │ Station  │                                       │   │
│   │      │ wanting  │                                       │   │
│   │      │ to send  │                                       │   │
│   │      └─────┬────┘                                       │   │
│   │            │                                             │   │
│   │            ▼                                             │   │
│   │      ┌──────────────────────────┐                       │   │
│   │      │ Is medium busy?          │                       │   │
│   │      └──────────┬───────────────┘                       │   │
│   │           │Yes        │No                               │   │
│   │           ▼           ▼                                 │   │
│   │        Wait       Start transmission                    │   │
│   │                                                          │   │
│   │   2. Multiple Access                                    │   │
│   │      Multiple stations can attempt to access medium     │   │
│   │                                                          │   │
│   │   3. Collision Detection                                │   │
│   │      ┌─────────────────────────────────────────────┐    │   │
│   │      │                                              │    │   │
│   │      │  Station A        Station B                 │    │   │
│   │      │       │                 │                   │    │   │
│   │      │       ▼                 ▼                   │    │   │
│   │      │  ══════════════◄►═══════════════           │    │   │
│   │      │          Collision occurred!                │    │   │
│   │      │                                              │    │   │
│   │      └─────────────────────────────────────────────┘    │   │
│   │                                                          │   │
│   │   4. Collision Handling (Binary Exponential Backoff)    │   │
│   │      - Send jam signal when collision detected          │   │
│   │      - Wait random time before retransmission           │   │
│   │      - Wait time = Random(0 ~ 2^n - 1) × Slot time      │   │
│   │      - n = collision count (max 10)                     │   │
│   │      - Give up after 16 collisions                      │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   CSMA/CD Flowchart:                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   [Prepare frame] → [Carrier sense]                     │   │
│   │                        │                                │   │
│   │              ┌─────────┼─────────┐                      │   │
│   │              │  Busy   │  Idle   │                      │   │
│   │              ▼         ▼                                │   │
│   │           [Wait]   [Start transmission]                 │   │
│   │                        │                                │   │
│   │              ┌─────────┼─────────┐                      │   │
│   │              │Collision│No collision                    │   │
│   │              ▼         ▼                                │   │
│   │         [Jam signal] [Transmission complete]            │   │
│   │              │                                          │   │
│   │              ▼                                          │   │
│   │         [Backoff]                                       │   │
│   │              │                                          │   │
│   │              └──────► [Carrier sense]                   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   * Modern full-duplex Ethernet does not require CSMA/CD        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Ethernet Standard Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                     Major Ethernet Standards                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   UTP-based:                                                    │
│   ┌────────────┬────────┬──────────┬────────┬────────────────┐  │
│   │  Standard  │  Speed │  Cable   │Distance│    Encoding    │  │
│   ├────────────┼────────┼──────────┼────────┼────────────────┤  │
│   │ 10BASE-T   │ 10Mbps │ Cat3/5   │  100m  │ Manchester     │  │
│   │ 100BASE-TX │ 100Mbps│ Cat5     │  100m  │ 4B/5B, MLT-3   │  │
│   │ 1000BASE-T │ 1Gbps  │ Cat5e    │  100m  │ 4D-PAM5        │  │
│   │ 10GBASE-T  │ 10Gbps │ Cat6a    │  100m  │ 128-DSQ        │  │
│   │ 25GBASE-T  │ 25Gbps │ Cat8     │  30m   │ PAM4           │  │
│   └────────────┴────────┴──────────┴────────┴────────────────┘  │
│                                                                  │
│   Fiber-based:                                                  │
│   ┌────────────┬────────┬──────────┬────────┬────────────────┐  │
│   │  Standard  │  Speed │  Fiber   │Distance│   Wavelength   │  │
│   ├────────────┼────────┼──────────┼────────┼────────────────┤  │
│   │ 100BASE-FX │ 100Mbps│ MMF      │  2km   │ 1310nm         │  │
│   │ 1000BASE-SX│ 1Gbps  │ MMF      │  550m  │ 850nm (Short)  │  │
│   │ 1000BASE-LX│ 1Gbps  │ SMF      │  10km  │ 1310nm (Long)  │  │
│   │ 10GBASE-SR │ 10Gbps │ MMF      │  300m  │ 850nm          │  │
│   │ 10GBASE-LR │ 10Gbps │ SMF      │  10km  │ 1310nm         │  │
│   │ 40GBASE-SR4│ 40Gbps │ MMF      │  100m  │ 850nm (4x10G)  │  │
│   │100GBASE-SR4│ 100Gbps│ MMF      │  100m  │ 850nm (4x25G)  │  │
│   └────────────┴────────┴──────────┴────────┴────────────────┘  │
│                                                                  │
│   * MMF = Multi-Mode Fiber, SMF = Single-Mode Fiber             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Switch Operation Principles

### Switch Functions

```
┌─────────────────────────────────────────────────────────────────┐
│                      Switch Functions                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Switch: Network device operating at L2 (Data Link) layer      │
│           Forwards frames based on MAC addresses                │
│                                                                  │
│   Hub vs Switch:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Hub:                                                  │   │
│   │   ┌───────────────────────────────────────────┐         │   │
│   │   │                                            │         │   │
│   │   │  PC1 ──┬                                  │         │   │
│   │   │        │    ┌─────┐                       │         │   │
│   │   │  PC2 ──┼────│ HUB │──── Copy to all ports│         │   │
│   │   │        │    └─────┘                       │         │   │
│   │   │  PC3 ──┘                                  │         │   │
│   │   │                                            │         │   │
│   │   │  - Forwards received frame to all ports   │         │   │
│   │   │  - Collision domain = entire (one domain) │         │   │
│   │   │                                            │         │   │
│   │   └───────────────────────────────────────────┘         │   │
│   │                                                          │   │
│   │   Switch:                                               │   │
│   │   ┌───────────────────────────────────────────┐         │   │
│   │   │                                            │         │   │
│   │   │  PC1 ──┬                                  │         │   │
│   │   │        │    ┌────────┐                    │         │   │
│   │   │  PC2 ──┼────│ SWITCH │──── Send to dest only│      │   │
│   │   │        │    └────────┘                    │         │   │
│   │   │  PC3 ──┘       │                          │         │   │
│   │   │                ▼                          │         │   │
│   │   │       Consult MAC table                   │         │   │
│   │   │                                            │         │   │
│   │   │  - Learns dest MAC and forwards to port only│       │   │
│   │   │  - Each port is separate collision domain  │         │   │
│   │   │                                            │         │   │
│   │   └───────────────────────────────────────────┘         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### MAC Address Table (CAM Table)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAC Address Table                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   CAM (Content Addressable Memory) Table:                       │
│   Mapping information between MAC addresses and port numbers    │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │              MAC Address Table                    │   │   │
│   │   ├─────────────────────────┬───────────┬───────────┤   │   │
│   │   │         MAC Address     │   Port    │   Type    │   │   │
│   │   ├─────────────────────────┼───────────┼───────────┤   │   │
│   │   │    00:1A:2B:3C:4D:5E   │    Fa0/1  │  Dynamic  │   │   │
│   │   │    00:1A:2B:3C:4D:5F   │    Fa0/2  │  Dynamic  │   │   │
│   │   │    00:1A:2B:3C:4D:60   │    Fa0/3  │  Static   │   │   │
│   │   │    00:1A:2B:3C:4D:61   │    Fa0/4  │  Dynamic  │   │   │
│   │   └─────────────────────────┴───────────┴───────────┘   │   │
│   │                                                          │   │
│   │   Types:                                                 │   │
│   │   - Dynamic: Added automatically via learning, expires  │   │
│   │   - Static: Manually configured, does not expire        │   │
│   │   - Default aging time: 300 seconds (5 minutes)         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   View MAC table (Cisco):                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   Switch# show mac address-table                         │   │
│   │                                                          │   │
│   │   Mac Address Table                                      │   │
│   │   -------------------------------------------            │   │
│   │   Vlan    Mac Address       Type        Ports            │   │
│   │   ----    -----------       --------    -----            │   │
│   │      1    001a.2b3c.4d5e    DYNAMIC     Fa0/1            │   │
│   │      1    001a.2b3c.4d5f    DYNAMIC     Fa0/2            │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Switch Learning and Forwarding

```
┌─────────────────────────────────────────────────────────────────┐
│                  Switch Learning and Forwarding Process          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Learning                                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   PC1 (MAC: AA) ──[Fa0/1]─── Switch                     │   │
│   │                                                          │   │
│   │   When PC1 transmits a frame:                           │   │
│   │   - Learn source MAC (AA) and receiving port (Fa0/1)    │   │
│   │   - Store in MAC table: AA → Fa0/1                      │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. Forwarding                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   When dest MAC is in table:                            │   │
│   │   - Forward frame only to that port (Unicast)           │   │
│   │                                                          │   │
│   │   PC1 (AA) ──[Fa0/1]──┬──[Fa0/2]── PC2 (BB)            │   │
│   │                       │                                 │   │
│   │                   Switch                                │   │
│   │                   BB → Fa0/2 (in table)                 │   │
│   │                       │                                 │   │
│   │                       └──[Fa0/3]── PC3 (CC)            │   │
│   │                          (no forwarding)                │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. Flooding                                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   When dest MAC is not in table:                        │   │
│   │   - Forward to all ports except receiving port          │   │
│   │                                                          │   │
│   │   PC1 (AA) ──[Fa0/1]──┬──[Fa0/2]── PC2 (BB) ←forward   │   │
│   │                       │                                 │   │
│   │                   Switch                                │   │
│   │                   XX → ? (not in table)                 │   │
│   │                       │                                 │   │
│   │                       └──[Fa0/3]── PC3 (CC) ←forward    │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   4. Filtering                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   When source and dest are on same port:                │   │
│   │   - Discard frame (no forwarding)                       │   │
│   │                                                          │   │
│   │   PC1 (AA) ──┬──[Fa0/1]── Switch                        │   │
│   │              │    │                                     │   │
│   │   PC2 (BB) ──┘    │  AA → BB frame                     │   │
│   │                   │  Both on Fa0/1 → discard            │   │
│   │                   │                                     │   │
│   │                   └── (no forwarding to other ports)    │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Switch Forwarding Methods

```
┌─────────────────────────────────────────────────────────────────┐
│                    Switch Forwarding Methods                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Store-and-Forward                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   [Receive frame] → [Store entire] → [CRC check] → [Forward]│
│   │                                                          │   │
│   │   - Receive entire frame then check FCS for errors      │   │
│   │   - Discard frames with errors                          │   │
│   │   - Most reliable, slight latency                       │   │
│   │   - Default method for modern switches                  │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. Cut-Through                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   [Check dest MAC] → [Start forwarding immediately]     │   │
│   │   (first 6 bytes only)                                  │   │
│   │                                                          │   │
│   │   - Check dest MAC only and forward immediately         │   │
│   │   - Lowest latency                                      │   │
│   │   - May forward errored frames                          │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. Fragment-Free                                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   [Receive first 64 bytes] → [Forward]                  │   │
│   │                                                          │   │
│   │   - Check up to minimum frame size (64 bytes) only      │   │
│   │   - Detects most collision-related errors               │   │
│   │   - Compromise between Cut-Through and Store-and-Forward│   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Comparison:                                                   │
│   ┌───────────────────┬──────────┬──────────┬────────────────┐  │
│   │      Method       │ Latency  │Error Det │     Use Case   │  │
│   ├───────────────────┼──────────┼──────────┼────────────────┤  │
│   │ Store-and-Forward │  High    │  Full    │ General        │  │
│   │ Cut-Through       │  Low     │  None    │ Low latency    │  │
│   │ Fragment-Free     │  Medium  │  Partial │ Compromise     │  │
│   └───────────────────┴──────────┴──────────┴────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. ARP (Address Resolution Protocol)

### Need for ARP

```
┌─────────────────────────────────────────────────────────────────┐
│                       Need for ARP                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Problem: IP communication requires destination MAC address    │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Host A                        Host B                  │   │
│   │   IP: 192.168.1.10               IP: 192.168.1.20       │   │
│   │   MAC: AA:AA:AA:AA:AA:AA         MAC: BB:BB:BB:BB:BB:BB │   │
│   │                                                          │   │
│   │        │                              │                 │   │
│   │        │ Send packet to 192.168.1.20  │                 │   │
│   │        │                              │                 │   │
│   │        │  Dest MAC = ???              │                 │   │
│   │        │                              │                 │   │
│   │        └─────────── ? ───────────────►│                 │   │
│   │                                                          │   │
│   │   Know IP address, but don't know MAC address!          │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Solution: Use ARP to resolve IP address → MAC address         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ARP Operation Process

```
┌─────────────────────────────────────────────────────────────────┐
│                      ARP Operation Process                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. ARP Request (Broadcast)                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Host A (192.168.1.10)                                 │   │
│   │       │                                                  │   │
│   │       │ "What is the MAC address of 192.168.1.20?"      │   │
│   │       │                                                  │   │
│   │       ▼ (Broadcast: FF:FF:FF:FF:FF:FF)                  │   │
│   │   ════════════════════════════════════════════          │   │
│   │       │             │              │                    │   │
│   │       ▼             ▼              ▼                    │   │
│   │   Host B        Host C         Host D                   │   │
│   │   (1.20)        (1.30)         (1.40)                   │   │
│   │   "My IP!"      "Not my IP"    "Not my IP"              │   │
│   │                 (ignore)        (ignore)                │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. ARP Reply (Unicast)                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Host B (192.168.1.20, MAC: BB:BB...)                 │   │
│   │       │                                                  │   │
│   │       │ "My MAC address is BB:BB:BB:BB:BB:BB"           │   │
│   │       │                                                  │   │
│   │       ▼ (Unicast: AA:AA:AA:AA:AA:AA)                    │   │
│   │   ────────────────────────────────────►                 │   │
│   │                                    │                     │   │
│   │                                    ▼                     │   │
│   │                               Host A                     │   │
│   │                               (Update ARP cache)         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. ARP Cache Storage                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Host A's ARP cache:                                   │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │  IP Address     │  MAC Address      │  Type    │   │   │
│   │   ├─────────────────┼───────────────────┼──────────┤   │   │
│   │   │  192.168.1.20   │  BB:BB:BB:BB:BB:BB│ Dynamic  │   │   │
│   │   │  192.168.1.1    │  CC:CC:CC:CC:CC:CC│ Dynamic  │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   │   - Use cache for subsequent communication (no re-ARP)  │   │
│   │   - Cache expiration time: typically 2-20 minutes       │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ARP Packet Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      ARP Packet Structure                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   0       8      16      24      32                      │   │
│   │   ├───────────────┬───────────────┤                     │   │
│   │   │ Hardware Type │ Protocol Type │                     │   │
│   │   │    (2B)       │     (2B)      │                     │   │
│   │   ├───────────────┼───────────────┤                     │   │
│   │   │ HW Addr Len   │ Proto Addr Len│   Opcode            │   │
│   │   │    (1B)       │     (1B)      │    (2B)             │   │
│   │   ├───────────────┴───────────────┴───────────────┤     │   │
│   │   │         Sender Hardware Address (6B)          │     │   │
│   │   ├───────────────────────────────────────────────┤     │   │
│   │   │         Sender Protocol Address (4B)          │     │   │
│   │   ├───────────────────────────────────────────────┤     │   │
│   │   │         Target Hardware Address (6B)          │     │   │
│   │   ├───────────────────────────────────────────────┤     │   │
│   │   │         Target Protocol Address (4B)          │     │   │
│   │   └───────────────────────────────────────────────┘     │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Field Descriptions:                                           │
│   ┌─────────────────────┬────────────────────────────────────┐  │
│   │ Hardware Type       │ 1 = Ethernet                       │  │
│   │ Protocol Type       │ 0x0800 = IPv4                      │  │
│   │ HW Addr Len         │ 6 (MAC address length)             │  │
│   │ Proto Addr Len      │ 4 (IPv4 address length)            │  │
│   │ Opcode              │ 1 = Request, 2 = Reply             │  │
│   │ Sender HW Addr      │ Sender MAC address                 │  │
│   │ Sender Proto Addr   │ Sender IP address                  │  │
│   │ Target HW Addr      │ Target MAC address (Request: 00:00)│  │
│   │ Target Proto Addr   │ Target IP address                  │  │
│   └─────────────────────┴────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ARP Cache Commands

```
┌─────────────────────────────────────────────────────────────────┐
│                      ARP Cache Commands                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   View ARP cache:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Windows:                                               │   │
│   │   C:\> arp -a                                            │   │
│   │   Interface: 192.168.1.100                               │   │
│   │     Internet Address    Physical Address    Type         │   │
│   │     192.168.1.1         aa-bb-cc-dd-ee-ff   dynamic     │   │
│   │     192.168.1.20        11-22-33-44-55-66   dynamic     │   │
│   │                                                          │   │
│   │   Linux/Mac:                                             │   │
│   │   $ arp -a                                               │   │
│   │   ? (192.168.1.1) at aa:bb:cc:dd:ee:ff on en0           │   │
│   │   ? (192.168.1.20) at 11:22:33:44:55:66 on en0          │   │
│   │                                                          │   │
│   │   Linux (ip command):                                    │   │
│   │   $ ip neigh show                                        │   │
│   │   192.168.1.1 dev eth0 lladdr aa:bb:cc:dd:ee:ff REACHABLE│   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Delete ARP cache:                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Windows:                                               │   │
│   │   C:\> arp -d *              (delete all)                │   │
│   │   C:\> arp -d 192.168.1.1    (specific entry)            │   │
│   │                                                          │   │
│   │   Linux:                                                 │   │
│   │   $ sudo arp -d 192.168.1.1                              │   │
│   │   $ sudo ip neigh del 192.168.1.1 dev eth0              │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Add static ARP:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Windows:                                               │   │
│   │   C:\> arp -s 192.168.1.50 aa-bb-cc-dd-ee-ff            │   │
│   │                                                          │   │
│   │   Linux:                                                 │   │
│   │   $ sudo arp -s 192.168.1.50 aa:bb:cc:dd:ee:ff          │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Collision Domain and Broadcast Domain

### Collision Domain

```
┌─────────────────────────────────────────────────────────────────┐
│                    Collision Domain                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Definition: Network area where collisions can occur when      │
│               transmitting simultaneously                        │
│                                                                  │
│   Hub's collision domain:                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │              One collision domain                 │   │   │
│   │   │                                                   │   │   │
│   │   │   PC1 ──┐                                        │   │   │
│   │   │         │                                        │   │   │
│   │   │   PC2 ──┼────[ HUB ]                            │   │   │
│   │   │         │                                        │   │   │
│   │   │   PC3 ──┘                                        │   │   │
│   │   │                                                   │   │   │
│   │   │   * All devices in same collision domain         │   │   │
│   │   │   * Collision occurs if PC1 and PC2 transmit     │   │   │
│   │   │     simultaneously                                │   │   │
│   │   │                                                   │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Switch's collision domain:                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌──────┐   ┌──────┐   ┌──────┐                       │   │
│   │   │ CD 1 │   │ CD 2 │   │ CD 3 │                       │   │
│   │   │      │   │      │   │      │                       │   │
│   │   │ PC1  │   │ PC2  │   │ PC3  │                       │   │
│   │   │  │   │   │  │   │   │  │   │                       │   │
│   │   └──┼───┘   └──┼───┘   └──┼───┘                       │   │
│   │      │          │          │                            │   │
│   │      └──────────┼──────────┘                            │   │
│   │                 │                                       │   │
│   │           [ SWITCH ]                                    │   │
│   │                                                          │   │
│   │   * Each port is separate collision domain              │   │
│   │   * PC1 and PC2 can transmit simultaneously (no collision)│ │
│   │   * No collisions at all when using full-duplex        │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Broadcast Domain

```
┌─────────────────────────────────────────────────────────────────┐
│                Broadcast Domain                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Definition: Network area where broadcast frames are delivered │
│                                                                  │
│   Switch's broadcast domain:                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │            One broadcast domain                   │   │   │
│   │   │                                                   │   │   │
│   │   │   PC1 ──┐                                        │   │   │
│   │   │         │                                        │   │   │
│   │   │   PC2 ──┼────[ SWITCH ]                         │   │   │
│   │   │         │                                        │   │   │
│   │   │   PC3 ──┘                                        │   │   │
│   │   │                                                   │   │   │
│   │   │   * Broadcasts forwarded to all ports            │   │   │
│   │   │   * Switches do not separate broadcast domains   │   │   │
│   │   │                                                   │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Router's broadcast domain:                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌────────────────────┐   ┌────────────────────┐       │   │
│   │   │   Broadcast        │   │   Broadcast        │       │   │
│   │   │   Domain 1         │   │   Domain 2         │       │   │
│   │   │                     │   │                     │       │   │
│   │   │   PC1 ──┐          │   │          ┌── PC3   │       │   │
│   │   │         │          │   │          │         │       │   │
│   │   │   PC2 ──┼──[SW1]   │   │   [SW2]──┼── PC4   │       │   │
│   │   │                     │   │                     │       │   │
│   │   └──────────┬──────────┘   └──────────┬─────────┘       │   │
│   │              │                          │                │   │
│   │              └────────[ ROUTER ]────────┘                │   │
│   │                                                          │   │
│   │   * Router does not forward broadcasts                  │   │
│   │   * Each interface is separate broadcast domain         │   │
│   │   * VLANs also separate broadcast domains               │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Device Domain Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    Device Domain Comparison                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────────┬─────────────────┬─────────────────────────┐ │
│   │     Device    │ Collision Domain│   Broadcast Domain      │ │
│   ├───────────────┼─────────────────┼─────────────────────────┤ │
│   │    Hub        │  No separation  │     No separation       │ │
│   │               │  (1 domain)     │     (1 domain)          │ │
│   ├───────────────┼─────────────────┼─────────────────────────┤ │
│   │   Bridge      │  Separates      │     No separation       │ │
│   │               │  (per port)     │     (1 domain)          │ │
│   ├───────────────┼─────────────────┼─────────────────────────┤ │
│   │   Switch      │  Separates      │     No separation       │ │
│   │               │  (per port)     │     (1 domain, exc VLAN)│ │
│   ├───────────────┼─────────────────┼─────────────────────────┤ │
│   │   Router      │  Separates      │     Separates           │ │
│   │               │  (per port)     │     (per interface)     │ │
│   └───────────────┴─────────────────┴─────────────────────────┘ │
│                                                                  │
│   Example network:                                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌──────────────────────────────────────────────────┐  │   │
│   │   │          Broadcast Domain 1                       │  │   │
│   │   │                                                   │  │   │
│   │   │   [PC1]──┐   ┌──[PC3]                           │  │   │
│   │   │   CD:1   │   │  CD:3                             │  │   │
│   │   │          │   │                                   │  │   │
│   │   │        [HUB]──[SWITCH]                          │  │   │
│   │   │          │      │                                │  │   │
│   │   │   [PC2]──┘      │                                │  │   │
│   │   │   CD:1          └─[PC4]                          │  │   │
│   │   │                    CD:4                          │  │   │
│   │   │                                                   │  │   │
│   │   └──────────────────────────────┬───────────────────┘  │   │
│   │                                  │                       │   │
│   │                             [ROUTER]                     │   │
│   │                                  │                       │   │
│   │   ┌──────────────────────────────┴───────────────────┐  │   │
│   │   │          Broadcast Domain 2                       │  │   │
│   │   │                                                   │  │   │
│   │   │        [SWITCH]                                   │  │   │
│   │   │          │                                        │  │   │
│   │   │   [PC5]──┴──[PC6]                                │  │   │
│   │   │   CD:5      CD:6                                  │  │   │
│   │   │                                                   │  │   │
│   │   └───────────────────────────────────────────────────┘  │   │
│   │                                                          │   │
│   │   Collision domains: 6 (HUB connections count as 1)     │   │
│   │   Broadcast domains: 2                                  │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Basic Problems

**1. Explain the structure of a MAC address and what OUI means.**

**2. Which of the following is a broadcast MAC address?**
   - (a) 00:00:00:00:00:00
   - (b) FF:FF:FF:FF:FF:FF
   - (c) 01:00:5E:00:00:01
   - (d) AA:BB:CC:DD:EE:FF

**3. How does a switch handle a frame with an unknown destination MAC address?**

**4. Explain why the minimum Ethernet frame size is 64 bytes in relation to CSMA/CD.**

### Applied Problems

**5. For the following network, determine the number of collision domains and broadcast domains.**

```
[PC1]──┬──[HUB]──┬──[PC2]
       │         │
       └──[SWITCH]──┬──[PC3]
                    │
               [ROUTER]
                    │
              [SWITCH]──┬──[PC4]
                        └──[PC5]
```

**6. Explain how ARP spoofing attacks work and suggest defense methods.**

**7. When PC A (192.168.1.10) transmits data to PC B (192.168.1.20) for the first time, explain step-by-step what happens from L2 and L3 perspectives.**

### Advanced Problems

**8. Explain the differences between Store-and-Forward and Cut-Through switching methods and the advantages/disadvantages of each.**

**9. Explain why CSMA/CD is not used when two PCs connected to the same switch communicate in Full-duplex mode.**

**10. Explain how VLANs separate broadcast domains.**

---

<details>
<summary>Answers</summary>

**1.**
- MAC address: 48 bits (6 bytes), hexadecimal notation
- First 3 bytes: OUI (Organizationally Unique Identifier) - Manufacturer identification
- Last 3 bytes: NIC unique ID - Assigned by manufacturer
- OUI is assigned by IEEE to each manufacturer

**2.** (b) FF:FF:FF:FF:FF:FF

**3.**
- Flooding: Forward frame to all ports except receiving port
- Learn source MAC address through replies

**4.**
- Must detect collision during transmission
- In worst case, transmission must continue for round-trip time (RTT) to detect collision
- Calculated at 10Mbps, 2.5km distance yields ~512 bits (64 bytes) required

**5.**
- Collision domains: 5 (HUB is 1, each switch port is 1)
- Broadcast domains: 2 (separated by router)

**6.**
- ARP spoofing: Attacker sends fake ARP Reply to manipulate other hosts' ARP cache
- Result: Traffic goes through attacker (man-in-the-middle attack)
- Defense: Static ARP entries, DAI (Dynamic ARP Inspection), 802.1X authentication

**7.**
1. PC A checks PC B's IP address
2. If not in ARP cache, broadcast ARP Request
3. PC B sends ARP Reply with MAC address
4. PC A updates ARP cache
5. Create Ethernet frame (Dest MAC: PC B, Source MAC: PC A)
6. Encapsulate IP packet in frame
7. Forward to PC B through switch

**8.**
- Store-and-Forward: Receive entire frame then check CRC, discard if error, higher latency
- Cut-Through: Check dest MAC only then forward immediately, lower latency, may forward errored frames
- Modern switches default to Store-and-Forward

**9.**
- Full-duplex: Separate circuits for transmit and receive
- Can transmit and receive simultaneously
- No collisions occur, so CSMA/CD unnecessary
- Each switch port is independent segment

**10.**
- VLAN: Divide one physical switch into multiple logical networks
- Broadcasts only delivered within same VLAN
- Communication between different VLANs requires router (L3)
- Reduces broadcast domain size, improves security

</details>

---

**Previous**: [Physical Layer](./04_Physical_Layer.md) | **Next**: [IP Addressing and Subnetting](./06_IP_Address_Subnetting.md)

---

## References

- Data Communications and Networking (Forouzan)
- [IEEE 802.3 Ethernet Standard](https://www.ieee802.org/3/)
- [RFC 826: ARP](https://tools.ietf.org/html/rfc826)
- [Cisco: Understanding Ethernet](https://www.cisco.com/c/en/us/support/docs/lan-switching/ethernet/20561-12.html)
