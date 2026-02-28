# 13. Zigbee and Z-Wave

**Previous**: [Cloud IoT Integration](./12_Cloud_IoT_Integration.md) | **Next**: [Multi-Sensor Fusion](./14_Multi_Sensor_Fusion.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare Zigbee, Z-Wave, Thread, and Matter protocols in terms of frequency, range, power, and mesh topology
2. Explain the IEEE 802.15.4 physical and MAC layer that underlies Zigbee and Thread
3. Describe Zigbee's mesh networking architecture including coordinators, routers, and end devices
4. Implement a simulated Zigbee mesh network with message routing and device management
5. Evaluate which wireless protocol best fits a given IoT use case

---

WiFi and Bluetooth serve personal devices well, but smart home and industrial IoT have different demands: hundreds of devices, years of battery life, and reliable coverage through walls. Zigbee and Z-Wave were designed specifically for these constraints. This lesson explores both protocols, compares them with the newer Thread/Matter standards, and shows how mesh networking enables large-scale IoT deployments.

---

## Table of Contents

1. [Low-Power Wireless Landscape](#1-low-power-wireless-landscape)
2. [IEEE 802.15.4 Foundation](#2-ieee-802154-foundation)
3. [Zigbee Protocol Stack](#3-zigbee-protocol-stack)
4. [Zigbee Mesh Networking](#4-zigbee-mesh-networking)
5. [Z-Wave Protocol](#5-z-wave-protocol)
6. [Thread and Matter](#6-thread-and-matter)
7. [Protocol Comparison](#7-protocol-comparison)
8. [Practical Applications](#8-practical-applications)
9. [Practice Problems](#9-practice-problems)

---

## 1. Low-Power Wireless Landscape

### 1.1 Why Not Just WiFi?

| Requirement | WiFi | BLE | Zigbee | Z-Wave |
|-------------|------|-----|--------|--------|
| Battery life | Hours-Days | Months | Years | Years |
| Max devices | ~32 | 7 (piconet) | 65,000 | 232 |
| Range (indoor) | 30-50m | 10-30m | 10-30m | 30-100m |
| Mesh support | No (AP-based) | Limited (BLE Mesh) | Native | Native |
| Data rate | 54-600 Mbps | 1-2 Mbps | 250 kbps | 100 kbps |
| Power (Tx) | 100-800 mW | 10-20 mW | 1-30 mW | 1-25 mW |

Smart home devices send tiny packets (a few bytes for "light on" or "temperature = 22.5°C") at low frequency (once per minute or less). High bandwidth is wasted; low power and reliable mesh coverage matter far more.

### 1.2 Frequency Bands

```
┌──────────────────────────────────────────────────────────┐
│                 ISM Frequency Bands                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Sub-GHz                                                 │
│  ├─ 868 MHz (EU) ─── Zigbee (optional), Z-Wave (EU)     │
│  └─ 908 MHz (US) ─── Z-Wave (US)                        │
│                                                          │
│  2.4 GHz (global)                                        │
│  ├─ WiFi (802.11b/g/n)                                   │
│  ├─ Bluetooth / BLE                                      │
│  ├─ Zigbee (802.15.4)                                    │
│  └─ Thread (802.15.4)                                    │
│                                                          │
│  Sub-GHz: Better wall penetration, longer range          │
│  2.4 GHz: Higher data rate, global availability          │
└──────────────────────────────────────────────────────────┘
```

---

## 2. IEEE 802.15.4 Foundation

IEEE 802.15.4 defines the physical (PHY) and medium access control (MAC) layers used by both Zigbee and Thread.

### 2.1 Physical Layer

| Parameter | Value |
|-----------|-------|
| Frequency | 2.4 GHz (16 channels), 868/915 MHz |
| Modulation | O-QPSK (2.4 GHz) |
| Data rate | 250 kbps (2.4 GHz) |
| Channel bandwidth | 2 MHz |
| Channels | 11-26 (2.4 GHz band) |

### 2.2 MAC Layer

The MAC layer uses CSMA-CA (Carrier Sense Multiple Access with Collision Avoidance):

```
Sender wants to transmit:
  1. Listen to channel (CCA = Clear Channel Assessment)
  2. If busy → random backoff, then retry
  3. If clear → transmit frame
  4. Wait for ACK from receiver
  5. No ACK? → retry (up to max_retries)
```

### 2.3 Frame Format

```
┌─────────┬──────────┬──────────┬──────────┬─────────┬──────┐
│ Preamble│ SFD      │ Frame    │ Frame    │ Payload │ FCS  │
│ (4 B)   │ (1 B)    │ Length   │ Control  │ (var)   │(2 B) │
│         │          │ (1 B)    │ (2 B)    │         │      │
└─────────┴──────────┴──────────┴──────────┴─────────┴──────┘
           Synchronization       MAC Header   Data    Checksum
```

Maximum payload: 127 bytes (small by design -- keeps frames short for reliability and low power).

---

## 3. Zigbee Protocol Stack

Zigbee adds network and application layers on top of IEEE 802.15.4:

```
┌─────────────────────────────────────┐
│  Application Layer                  │
│  ├── ZCL (Zigbee Cluster Library)   │  Standard device profiles
│  └── ZDO (Zigbee Device Object)     │  Device discovery, management
├─────────────────────────────────────┤
│  Application Support (APS)          │  Binding, group management
├─────────────────────────────────────┤
│  Network Layer (NWK)                │  Mesh routing, security
├─────────────────────────────────────┤
│  MAC Layer (IEEE 802.15.4)          │  CSMA-CA, framing
├─────────────────────────────────────┤
│  Physical Layer (IEEE 802.15.4)     │  2.4 GHz radio
└─────────────────────────────────────┘
```

### 3.1 Zigbee Cluster Library (ZCL)

ZCL defines standard "clusters" -- reusable functional blocks that ensure interoperability:

| Cluster | ID | Functions |
|---------|-----|----------|
| On/Off | 0x0006 | Turn device on/off, toggle |
| Level Control | 0x0008 | Brightness dimming (0-255) |
| Color Control | 0x0300 | Hue, saturation, color temperature |
| Temperature | 0x0402 | Read temperature sensor |
| Occupancy | 0x0406 | Motion detection |
| Door Lock | 0x0101 | Lock/unlock, PIN management |
| IAS Zone | 0x0500 | Intrusion/alarm sensors |

A Zigbee light bulb implements the On/Off and Level Control clusters; a thermostat implements Temperature Measurement. Any controller that speaks these clusters can operate any manufacturer's device.

---

## 4. Zigbee Mesh Networking

### 4.1 Device Roles

| Role | Description | Power | Examples |
|------|-------------|-------|---------|
| **Coordinator** | Forms the network, assigns addresses, one per network | Mains-powered | Hub, gateway |
| **Router** | Forwards messages, extends range | Mains-powered | Smart plugs, light bulbs |
| **End Device** | Sleeps to save power, communicates through parent router | Battery | Sensors, buttons |

### 4.2 Network Formation

```
1. Coordinator starts:
   - Scans channels for least interference
   - Selects PAN ID (16-bit)
   - Begins accepting join requests

2. Router joins:
   - Scans for coordinator's beacon
   - Sends Association Request
   - Receives 16-bit network address
   - Becomes available as parent for end devices

3. End device joins:
   - Finds nearest router (strongest signal)
   - Associates with router as parent
   - Sleeps most of the time
   - Wakes periodically to poll parent for pending messages
```

### 4.3 Mesh Routing

Zigbee uses AODV (Ad-hoc On-demand Distance Vector) routing:

```
Source → Router A → Router B → Router C → Destination

1. Source broadcasts Route Request (RREQ)
2. Intermediate routers forward RREQ, recording reverse path
3. Destination sends Route Reply (RREP) back along reverse path
4. Source caches the discovered route
5. Data packets follow the cached route
6. If a link fails → Route Error (RERR) triggers rediscovery
```

### 4.4 Self-Healing

When a router fails, the mesh automatically reroutes:

```
Before failure:              After Router B fails:
A ─── B ─── C               A ─── D ─── C
│     │     │                │           │
D ─── E ─── F               │     E ─── F
                             └─────┘
                             (new route discovered via D)
```

---

## 5. Z-Wave Protocol

### 5.1 Key Differences from Zigbee

| Feature | Zigbee | Z-Wave |
|---------|--------|--------|
| Standard body | Zigbee Alliance (CSA) | Z-Wave Alliance (Silicon Labs) |
| Frequency | 2.4 GHz (global) | Sub-GHz (region-specific) |
| Data rate | 250 kbps | 100 kbps (9.6/40/100) |
| Range | 10-30m | 30-100m (sub-GHz advantage) |
| Max devices | 65,000 | 232 |
| Chip vendors | Multiple (TI, NXP, Silicon Labs) | Silicon Labs only |
| Interoperability | Cluster-based (sometimes issues) | Mandatory certification |

### 5.2 Z-Wave Advantages

- **Sub-GHz frequency**: Better wall penetration and longer range than 2.4 GHz
- **Mandatory certification**: Every Z-Wave device must pass interoperability testing, reducing "works on paper but not in practice" issues
- **Source routing**: Z-Wave uses source routing (the sender specifies the full path), which reduces routing table memory on constrained devices

### 5.3 Z-Wave Mesh Topology

```
┌─────────────────────────────────────────────────┐
│               Z-Wave Network                     │
│                                                  │
│   Controller (1)  ←── Primary controller         │
│       │                manages routing table      │
│   ┌───┼───┐                                      │
│   │   │   │                                      │
│   R1  R2  R3      ←── Routing slaves (always on) │
│   │   │   │                                      │
│   S1  S2  S3      ←── Slaves (sensors, battery)  │
│                                                  │
│   Max 4 hops between any two devices             │
└─────────────────────────────────────────────────┘
```

---

## 6. Thread and Matter

### 6.1 Thread

Thread is a newer mesh networking protocol (2014) built on IEEE 802.15.4, designed to address Zigbee's limitations:

| Feature | Thread | Zigbee |
|---------|--------|--------|
| IP-based | Yes (IPv6 / 6LoWPAN) | No (custom addressing) |
| Cloud integration | Native (IP routing) | Requires gateway translation |
| Border Router | Any device can be one | Single coordinator |
| Single point of failure | No (multiple BRs) | Yes (coordinator) |
| Standard | Open (IETF standards) | Proprietary application layer |

### 6.2 Matter (formerly CHIP)

Matter is an application-layer protocol that runs over Thread, WiFi, or Ethernet:

```
┌─────────────────────────────────────────────────┐
│                    Matter                        │
│            (Application Layer)                   │
│                                                  │
│  Unified device models, setup, and control       │
│  Backed by Apple, Google, Amazon, Samsung        │
├──────────┬──────────┬──────────┬────────────────┤
│  Thread  │   WiFi   │ Ethernet │   BLE          │
│  (mesh)  │ (high BW)│ (wired)  │ (commissioning)│
└──────────┴──────────┴──────────┴────────────────┘
```

Matter aims to solve the fragmentation problem: one standard that works with HomeKit, Google Home, Alexa, and SmartThings simultaneously.

---

## 7. Protocol Comparison

### 7.1 Decision Matrix

| Use Case | Best Protocol | Why |
|----------|--------------|-----|
| Smart home (new) | **Matter** over Thread | Future-proof, multi-ecosystem |
| Smart home (existing) | **Z-Wave** | Proven reliability, long range |
| Large building | **Zigbee** | 65K device limit, ZCL profiles |
| Industrial monitoring | **Zigbee** or **Thread** | Mesh self-healing, IP-based |
| Battery sensors | **Z-Wave** or **Zigbee** | Sub-1mA sleep current |
| Outdoor/long range | **LoRaWAN** | Km-range, not covered here |

### 7.2 Range and Penetration

```
Indoor Range (typical):

Z-Wave (908 MHz):  ████████████████████████████████  30-100m
Zigbee (2.4 GHz):  ████████████████                  10-30m
BLE (2.4 GHz):     ████████████                      10-20m
WiFi (2.4 GHz):    ████████████████████              20-50m

Wall penetration (relative):
Z-Wave:  ★★★★★  (sub-GHz penetrates better)
Zigbee:  ★★★    (2.4 GHz reflected more by walls)
WiFi:    ★★★    (similar to Zigbee)
BLE:     ★★     (lower transmit power)
```

---

## 8. Practical Applications

### 8.1 Smart Home System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Smart Home                          │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Cloud/App   │◄──►│  Hub/Border Router       │   │
│  │ (HomeKit,    │    │  (Matter controller)      │   │
│  │  Google Home)│    └──────────┬───────────────┘   │
│  └──────────────┘               │                    │
│                    ┌────────────┼────────────┐       │
│                    │            │            │       │
│              ┌─────┴─────┐ ┌───┴────┐ ┌────┴────┐  │
│              │Thread Mesh│ │Z-Wave  │ │WiFi     │  │
│              │           │ │Mesh    │ │Devices  │  │
│              │• Sensors  │ │• Locks │ │• Cameras│  │
│              │• Lights   │ │• Plugs │ │• Speakers│ │
│              │• Switches │ │• Blinds│ │         │  │
│              └───────────┘ └────────┘ └─────────┘  │
└─────────────────────────────────────────────────────┘
```

### 8.2 Industrial IoT Monitoring

Zigbee networks in factories monitor hundreds of sensors:

- Temperature/humidity across production floor (100+ nodes)
- Vibration sensors on machinery (predictive maintenance)
- Air quality monitors (safety compliance)
- Asset tracking tags (inventory management)

The mesh topology handles the harsh RF environment (metal structures, interference) through path diversity and self-healing.

---

## 9. Practice Problems

### Problem 1: Protocol Selection

For each scenario, choose the most appropriate protocol and justify your choice:
1. A 200-room hotel wants wireless door locks with 5-year battery life
2. A greenhouse needs 500 soil moisture sensors covering 10 acres
3. A homeowner wants to control 15 smart bulbs, 3 thermostats, and 2 door locks
4. A factory needs real-time vibration monitoring on 50 machines with <100ms latency

### Problem 2: Mesh Network Design

Design a Zigbee mesh network for a two-story house (150 m² per floor):
1. Determine the minimum number of routers needed for full coverage
2. Draw the network topology
3. Show the routing path from a basement sensor to a second-floor display
4. Explain what happens when the ground-floor router closest to the stairs fails

### Problem 3: Power Budget Analysis

A Zigbee end device (battery-powered temperature sensor) has:
- Battery: 2× AA (3000 mAh total at 3V)
- Sleep current: 5 μA
- Transmit current: 20 mA
- Transmit time per reading: 15 ms
- Reading frequency: once per 5 minutes

Calculate the expected battery life in years. What is the dominant power consumer?

### Problem 4: Z-Wave vs Zigbee Range

A home has concrete walls that attenuate signals by 12 dB per wall. The minimum signal strength for reliable reception is -95 dBm.
1. If a Zigbee device transmits at 0 dBm and free-space path loss at 10m is -60 dBm, how many concrete walls can the signal penetrate?
2. If a Z-Wave device transmits at -2 dBm but has -5 dBm better receiver sensitivity (-100 dBm), how many walls can it penetrate?
3. How does mesh routing change this analysis?

### Problem 5: Thread Network Simulation

Implement a simple Thread-like mesh network simulator in Python:
1. Create 20 nodes at random positions in a 100×100m area
2. Two nodes are designated as Border Routers
3. Find the shortest path (hop count) between any end device and the nearest Border Router
4. Simulate removing a random node and show that the network self-heals (finds alternative paths)

---

*End of Lesson 13*
