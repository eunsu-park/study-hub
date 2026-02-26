# Physical Layer

**Previous**: [TCP/IP Model](./03_TCP_IP_Model.md) | **Next**: [Data Link Layer](./05_Data_Link_Layer.md)

**Difficulty**: ⭐ (Beginner)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of the Physical Layer in converting bits to physical signals
2. Compare guided (wired) and unguided (wireless) transmission media
3. Distinguish between UTP, STP, and fiber optic cables and select the right one for a given scenario
4. Explain the difference between analog and digital signals and describe common modulation techniques
5. Calculate channel capacity using the Nyquist and Shannon formulas
6. Identify Ethernet cable categories (Cat5e through Cat8) and their speed/distance specifications
7. Compare Wi-Fi standards (802.11n/ac/ax) and frequency bands (2.4 GHz vs 5 GHz)

---

The Physical Layer is where networking meets the real world. No matter how elegant the protocols above it, every bit of data ultimately becomes an electrical pulse on a copper wire, a flash of light in a fiber strand, or a radio wave through the air. Choosing the right cable, understanding signal degradation, and knowing the limits of your transmission medium directly determines network speed, reliability, and reach.

## Table of Contents

1. [Role of the Physical Layer](#1-role-of-the-physical-layer)
2. [Transmission Media](#2-transmission-media)
3. [Signal Types](#3-signal-types)
4. [Bandwidth and Transmission Speed](#4-bandwidth-and-transmission-speed)
5. [Ethernet Cable Types](#5-ethernet-cable-types)
6. [Connector Types](#6-connector-types)
7. [Wireless Transmission](#7-wireless-transmission)
8. [Practice Problems](#8-practice-problems)

---

## 1. Role of the Physical Layer

### Definition of Physical Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                     Physical Layer                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   "The layer that converts bit streams into physical signals    │
│    and actually transmits them through transmission media"      │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Digital Data (Upper layers)                           │   │
│   │         │                                                │   │
│   │         ▼                                                │   │
│   │   ┌─────────────────────────────────────┐               │   │
│   │   │        Physical Layer                │               │   │
│   │   │                                       │               │   │
│   │   │   Bits → Signal Conversion           │               │   │
│   │   │   (Encoding/Modulation)              │               │   │
│   │   │                                       │               │   │
│   │   └───────────────┬─────────────────────┘               │   │
│   │                   │                                      │   │
│   │                   ▼                                      │   │
│   │   ┌─────────────────────────────────────┐               │   │
│   │   │         Transmission Media           │               │   │
│   │   │   (Cable, Fiber, Wireless)           │               │   │
│   │   └─────────────────────────────────────┘               │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Major Functions of Physical Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                    Physical Layer Functions                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Bit Synchronization                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   Synchronizes sender and receiver clocks to accurately │   │
│   │   recognize bits                                         │   │
│   │                                                          │   │
│   │   TX Clock:  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐                       │   │
│   │             └─┘ └─┘ └─┘ └─┘ └─┘                       │   │
│   │   RX Clock:  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐  (Sync required)     │   │
│   │             └─┘ └─┘ └─┘ └─┘ └─┘                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. Bit Rate Control                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   Determines bits per second (bps) transmitted          │   │
│   │   Examples: 100 Mbps, 1 Gbps, 10 Gbps                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. Physical Topology Definition                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   Physical connection method between devices             │   │
│   │   (Bus, Star, Ring, etc.)                               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   4. Transmission Mode Definition                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Simplex:                                              │   │
│   │   A ───────────────────────────────► B                  │   │
│   │   (Keyboard → Computer, TV broadcast)                   │   │
│   │                                                          │   │
│   │   Half-Duplex:                                          │   │
│   │   A ◄──────────────────────────────► B                  │   │
│   │   (One direction at a time, Walkie-talkie)              │   │
│   │                                                          │   │
│   │   Full-Duplex:                                          │   │
│   │   A ◄═══════════════════════════════► B                  │   │
│   │   (Both directions simultaneously, Phone, Ethernet)     │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   5. Interface and Media Definition                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   - Connector types (RJ-45, SFP, etc.)                  │   │
│   │   - Cable types (UTP, Fiber, etc.)                      │   │
│   │   - Voltage levels, Signal timing                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Physical Layer Equipment

```
┌─────────────────────────────────────────────────────────────────┐
│                    Physical Layer Equipment                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┬───────────────────────────────────────────┐  │
│   │   Device     │                Function                    │  │
│   ├──────────────┼───────────────────────────────────────────┤  │
│   │    Hub       │ Multi-port repeater, signal amplification │  │
│   │              │ Broadcasts to all ports                   │  │
│   ├──────────────┼───────────────────────────────────────────┤  │
│   │   Repeater   │ Signal amplification, extends distance    │  │
│   │              │ Restores attenuated signals               │  │
│   ├──────────────┼───────────────────────────────────────────┤  │
│   │    NIC       │ Network Interface Card                    │  │
│   │(Network Card)│ Connects computer to network              │  │
│   ├──────────────┼───────────────────────────────────────────┤  │
│   │    Modem     │ Digital ↔ Analog signal conversion        │  │
│   │              │ Data transmission over phone lines        │  │
│   ├──────────────┼───────────────────────────────────────────┤  │
│   │  Media       │ Connects different media types            │  │
│   │  Converter   │ Example: Fiber ↔ UTP conversion           │  │
│   └──────────────┴───────────────────────────────────────────┘  │
│                                                                  │
│   Hub Operation:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   PC1 ────┐                                             │   │
│   │           │                                             │   │
│   │   PC2 ────┼──── HUB ───► Copies signal to all ports    │   │
│   │           │                                             │   │
│   │   PC3 ────┘                                             │   │
│   │                                                          │   │
│   │   * Hub is L1 device, doesn't understand MAC addresses  │   │
│   │   * Forwards received signal to all ports (broadcast)   │   │
│   │   * Collision possible (single collision domain)        │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Transmission Media

### Transmission Media Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                  Transmission Media Classification               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                        Transmission Media                        │
│                           │                                      │
│           ┌───────────────┴───────────────┐                     │
│           │                               │                     │
│       Guided Media                    Unguided Media            │
│       (Wired)                         (Wireless)                │
│           │                               │                     │
│   ┌───────┼───────┐               ┌───────┼───────┐            │
│   │       │       │               │       │       │            │
│ Coaxial  Twisted  Fiber           Radio  Micro-   Infrared     │
│  Cable    Pair    Optic           Waves   wave                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Wired Transmission Media

#### Coaxial Cable

```
┌─────────────────────────────────────────────────────────────────┐
│                   Coaxial Cable                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Structure:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │                  ┌─── Jacket (Outer)                    │   │
│   │                  │                                       │   │
│   │   ╔══════════════════════════════════════════════╗      │   │
│   │   ║┌────────────────────────────────────────────┐║      │   │
│   │   ║│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│║←Shield│   │
│   │   ║│  ┌────────────────────────────────────┐   │║      │   │
│   │   ║│  │ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ │   │║←Insulation│
│   │   ║│  │                                    │   │║      │   │
│   │   ║│  │      ═══════════════════════       │   │║←Center│   │
│   │   ║│  │           (Copper)                  │   │║Conductor│
│   │   ║│  │                                    │   │║      │   │
│   │   ║│  └────────────────────────────────────┘   │║      │   │
│   │   ║└────────────────────────────────────────────┘║      │   │
│   │   ╚══════════════════════════════════════════════╝      │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Types:                                                        │
│   ┌──────────────┬───────────┬───────────┬────────────────┐    │
│   │    Type      │ Impedance │   Usage   │   Connector    │    │
│   ├──────────────┼───────────┼───────────┼────────────────┤    │
│   │  Thin        │    50Ω    │ Legacy LAN│    BNC         │    │
│   │              │           │           │                │    │
│   ├──────────────┼───────────┼───────────┼────────────────┤    │
│   │  Thick       │    50Ω    │ Backbone  │    BNC         │    │
│   │              │           │           │                │    │
│   ├──────────────┼───────────┼───────────┼────────────────┤    │
│   │  RG-6        │    75Ω    │ Cable TV  │    F-Type      │    │
│   │              │           │ CATV      │                │    │
│   └──────────────┴───────────┴───────────┴────────────────┘    │
│                                                                  │
│   Characteristics:                                              │
│   - Strong resistance to external interference (shielding)      │
│   - Currently mainly used for cable TV, CCTV                    │
│   - Mostly replaced by UTP in LANs                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Twisted Pair

```
┌─────────────────────────────────────────────────────────────────┐
│                  Twisted Pair                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Structure (Twisted wires):                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   4 pairs of twisted copper wires (total 8 strands)     │   │
│   │                                                          │   │
│   │         ╭─╮   ╭─╮   ╭─╮   ╭─╮                          │   │
│   │        ╱   ╲ ╱   ╲ ╱   ╲ ╱   ╲  ← Pair 1 (Orange)     │   │
│   │       ╱     ╳     ╳     ╳     ╲                        │   │
│   │      ╱     ╱ ╲   ╱ ╲   ╱ ╲                              │   │
│   │     ╭─╮   ╭─╮   ╭─╮   ╭─╮                              │   │
│   │                                                          │   │
│   │   ※ Twisting cancels electromagnetic interference (EMI)│   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Types:                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   UTP (Unshielded Twisted Pair) - Unshielded           │   │
│   │   ┌────────────────────────────────────────┐            │   │
│   │   │  ┌──────────────────────────────────┐  │            │   │
│   │   │  │  ∞∞∞∞∞  ∞∞∞∞∞  ∞∞∞∞∞  ∞∞∞∞∞ │  │←Jacket     │   │
│   │   │  │  (Pair1) (Pair2) (Pair3) (Pair4)│  │            │   │
│   │   │  └──────────────────────────────────┘  │            │   │
│   │   └────────────────────────────────────────┘            │   │
│   │   - Most common, inexpensive                            │   │
│   │   - Office, home LANs                                   │   │
│   │                                                          │   │
│   │   STP (Shielded Twisted Pair) - Shielded               │   │
│   │   ┌────────────────────────────────────────┐            │   │
│   │   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│←Shield    │   │
│   │   │  ┌──────────────────────────────────┐  │            │   │
│   │   │  │  ∞∞∞∞∞  ∞∞∞∞∞  ∞∞∞∞∞  ∞∞∞∞∞ │  │            │   │
│   │   │  └──────────────────────────────────┘  │            │   │
│   │   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │   │
│   │   └────────────────────────────────────────┘            │   │
│   │   - EMI shielding, expensive                            │   │
│   │   - Industrial, high-interference environments          │   │
│   │                                                          │   │
│   │   FTP (Foiled Twisted Pair) - Foil shielded            │   │
│   │   - Wrapped in foil                                     │   │
│   │   - Between UTP and STP                                 │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Fiber Optic

```
┌─────────────────────────────────────────────────────────────────┐
│                      Fiber Optic                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Principle: Data transmission using total internal reflection  │
│                                                                  │
│   Structure:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │         ┌─── Jacket (Outer)                             │   │
│   │         │                                                │   │
│   │   ╔═════════════════════════════════════════════════╗   │   │
│   │   ║  ┌─────────────────────────────────────────────┐║   │   │
│   │   ║  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│║←Buffer│   │
│   │   ║  │  ┌───────────────────────────────────────┐ │║   │   │
│   │   ║  │  │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ │║←Cladding│
│   │   ║  │  │  ┌─────────────────────────────────┐ │ │║   │   │
│   │   ║  │  │  │═════════════════════════════════│ │ │║←Core │   │
│   │   ║  │  │  │       (Glass/Plastic)            │ │ │║(Light path)│
│   │   ║  │  │  └─────────────────────────────────┘ │ │║   │   │
│   │   ║  │  └───────────────────────────────────────┘ │║   │   │
│   │   ║  └─────────────────────────────────────────────┘║   │   │
│   │   ╚═════════════════════════════════════════════════╝   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Types:                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Single-Mode Fiber (SMF)                               │   │
│   │   ┌──────────────────────────────────────────────────┐  │   │
│   │   │     Core diameter: 8-10 μm                        │  │   │
│   │   │                                                   │  │   │
│   │   │     ════════════════════════════════► (Single path)│  │   │
│   │   │                                                   │  │   │
│   │   │     - Long distance (tens~hundreds of km)         │  │   │
│   │   │     - High bandwidth                              │  │   │
│   │   │     - Uses laser light source                     │  │   │
│   │   │     - Yellow cable (typically)                    │  │   │
│   │   └──────────────────────────────────────────────────┘  │   │
│   │                                                          │   │
│   │   Multi-Mode Fiber (MMF)                                │   │
│   │   ┌──────────────────────────────────────────────────┐  │   │
│   │   │     Core diameter: 50-62.5 μm                     │  │   │
│   │   │                                                   │  │   │
│   │   │     ══════════════════════════════►               │  │   │
│   │   │        ═══════════════════════════► (Multiple paths)│  │   │
│   │   │     ══════════════════════════════►               │  │   │
│   │   │                                                   │  │   │
│   │   │     - Short distance (hundreds m ~ 2 km)          │  │   │
│   │   │     - LED or VCSEL light source                   │  │   │
│   │   │     - Orange/Aqua cable                           │  │   │
│   │   │     - Data center internal connections            │  │   │
│   │   └──────────────────────────────────────────────────┘  │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Advantages:                                                   │
│   - Complete immunity to electromagnetic interference (EMI)     │
│   - Very high bandwidth (terabit-class)                         │
│   - Long distance transmission capable                          │
│   - High security (difficult to tap)                            │
│                                                                  │
│   Disadvantages:                                                │
│   - High installation cost                                      │
│   - Difficult connection/repair (requires special equipment)    │
│   - Vulnerable to bending                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Transmission Media Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                 Transmission Media Comparison                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┬──────────┬──────────┬─────────┬─────────────────┐ │
│   │ Media   │ Bandwidth│ Distance │ Cost    │ EMI Immunity    │ │
│   ├─────────┼──────────┼──────────┼─────────┼─────────────────┤ │
│   │ Coaxial │ ~1 Gbps  │ ~500m    │ Medium  │ Good            │ │
│   │ Cable   │          │          │         │                 │ │
│   ├─────────┼──────────┼──────────┼─────────┼─────────────────┤ │
│   │ UTP     │ ~10 Gbps │ 100m     │ Low     │ Fair            │ │
│   │         │          │          │         │                 │ │
│   ├─────────┼──────────┼──────────┼─────────┼─────────────────┤ │
│   │ STP     │ ~10 Gbps │ 100m     │ Medium  │ Good            │ │
│   │         │          │          │         │                 │ │
│   ├─────────┼──────────┼──────────┼─────────┼─────────────────┤ │
│   │ MMF     │ ~100 Gbps│ ~2km     │ High    │ Immune          │ │
│   │ (Fiber) │          │          │         │                 │ │
│   ├─────────┼──────────┼──────────┼─────────┼─────────────────┤ │
│   │ SMF     │ Terabit  │ ~100km   │ Very    │ Immune          │ │
│   │ (Fiber) │          │          │ High    │                 │ │
│   └─────────┴──────────┴──────────┴─────────┴─────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Signal Types

### Analog vs Digital Signals

```
┌─────────────────────────────────────────────────────────────────┐
│                    Analog vs Digital Signals                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Analog Signal:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Voltage ^                                              │   │
│   │       │      ╭───╮        ╭───╮        ╭───╮           │   │
│   │       │    ╱      ╲      ╱      ╲      ╱      ╲          │   │
│   │    0 ─┼───────────────────────────────────────── Time   │   │
│   │       │  ╲      ╱      ╲      ╱      ╲      ╱           │   │
│   │       │    ╰───╯        ╰───╯        ╰───╯              │   │
│   │                                                          │   │
│   │   - Continuous waveform                                 │   │
│   │   - Amplitude, frequency, phase vary continuously       │   │
│   │   - Examples: Voice, radio broadcast, legacy phones     │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Digital Signal:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Voltage ^                                              │   │
│   │       │   ┌───┐       ┌───┐   ┌───────┐   ┌───┐        │   │
│   │    1 ─┤   │   │       │   │   │       │   │   │        │   │
│   │       │   │   │       │   │   │       │   │   │        │   │
│   │    0 ─┼───┘   └───────┘   └───┘       └───┘   └──── Time│   │
│   │       │                                                  │   │
│   │       │   1   0   0   1   0   1   1   1   0   1         │   │
│   │                                                          │   │
│   │   - Discrete values (0 or 1)                            │   │
│   │   - Resistant to noise (clear threshold)                │   │
│   │   - Examples: Computer data, Ethernet                   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Signal Conversion

> **Why do different encoding schemes exist?** The core problem is: how do you represent 0s and 1s on a wire? **NRZ** is simplest (high=1, low=0) but the receiver loses clock sync during long runs of the same bit. **Manchester** solves this by guaranteeing a transition every bit period — the price is double the bandwidth. **4B/5B** strikes a balance: encode 4 data bits as 5 wire bits to limit consecutive identical symbols, then use NRZ for transmission. Each scheme trades bandwidth efficiency for clock recovery reliability. **Analog modulation** (ASK/FSK/PSK/QAM) solves a different problem: transmitting digital data over analog media (radio, phone lines) by encoding bits into carrier wave properties.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Signal Conversion                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Digital-Digital Encoding (Line Encoding)                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   NRZ (Non-Return to Zero):                             │   │
│   │        ┌───┐   ┌───────┐       ┌───┐                    │   │
│   │     1  │   │   │       │       │   │                    │   │
│   │     0 ─┘   └───┘       └───────┘   └───                 │   │
│   │        1   0   1   1   0   0   0   1                     │   │
│   │                                                          │   │
│   │   Manchester:                                           │   │
│   │        ┌─┐ ┌─┐ ┌─┐ ┌─┐   ┌─┐   ┌─┐ ┌─┐                 │   │
│   │     1  │ │ │ │ │ │ │ │   │ │   │ │ │ │                 │   │
│   │     0 ─┘ └─┘ └─┘ └─┘ └───┘ └───┘ └─┘ └─                │   │
│   │   - Transition in middle of bit (clock synchronization) │   │
│   │   - Used in Ethernet 10BASE-T                           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. Digital-Analog Modulation                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ASK (Amplitude Shift Keying) - Amplitude modulation:  │   │
│   │   Data:  1    0    1    1    0                          │   │
│   │          ∿∿∿       ∿∿∿  ∿∿∿                              │   │
│   │                                                          │   │
│   │   FSK (Frequency Shift Keying) - Frequency modulation:  │   │
│   │   Data:  1    0    1    1    0                          │   │
│   │          ∿∿∿  ≋≋≋  ∿∿∿  ∿∿∿  ≋≋≋                       │   │
│   │          (High freq) (Low freq)                         │   │
│   │                                                          │   │
│   │   PSK (Phase Shift Keying) - Phase modulation:          │   │
│   │   - Changes phase to represent data                     │   │
│   │   - Used in Wi-Fi, satellite communications             │   │
│   │                                                          │   │
│   │   QAM (Quadrature Amplitude Modulation):                │   │
│   │   - Modulates both amplitude and phase                  │   │
│   │   - High bandwidth efficiency                           │   │
│   │   - Used in cable modems, DSL                           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. Analog-Digital Conversion (A/D Conversion)                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Voice/Video → Sampling → Quantization → Encoding      │   │
│   │   → Digital Data                                        │   │
│   │                                                          │   │
│   │   Example: Phone voice (PCM - Pulse Code Modulation)    │   │
│   │   - 8,000 samples per second                            │   │
│   │   - 8-bit quantization                                  │   │
│   │   - 64 Kbps (8000 × 8 = 64,000)                         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Signal Impairment Factors

```
┌─────────────────────────────────────────────────────────────────┐
│                     Signal Impairment Factors                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Attenuation                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   TX Signal:  ████████                                  │   │
│   │                  ↓ (Weakens with distance)               │   │
│   │   RX Signal:     ▓▓▓▓                                   │   │
│   │                                                          │   │
│   │   Solution: Use repeaters, amplifiers                   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. Distortion                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Original:  ┌─┐   ┌─┐                                  │   │
│   │              │ │   │ │                                  │   │
│   │              └─┘   └─┘                                  │   │
│   │                  ↓                                       │   │
│   │   Distorted: ╱─╲   ╱─╲                                  │   │
│   │              ╱   ╲ ╱   ╲                                │   │
│   │                                                          │   │
│   │   - Different frequency components travel at different  │   │
│   │     speeds                                               │   │
│   │   - Corrected with equalizers                           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. Noise                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Original:  ┌─┐   ┌─┐                                  │   │
│   │              │ │   │ │                                  │   │
│   │              └─┘   └─┘                                  │   │
│   │                  ↓ + Noise                               │   │
│   │   Received:  ┌∿┐ ∿ ┌∿┐                                  │   │
│   │              │ │∿  │ │                                  │   │
│   │              └∿┘   └∿┘                                  │   │
│   │                                                          │   │
│   │   Types:                                                 │   │
│   │   - Thermal Noise                                       │   │
│   │   - Induced Noise - EMI                                 │   │
│   │   - Crosstalk - Adjacent line interference              │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Bandwidth and Transmission Speed

### Bandwidth Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                     Bandwidth                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Two meanings:                                                 │
│                                                                  │
│   1. Analog Bandwidth (Hz)                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Frequency range used by signal                        │   │
│   │                                                          │   │
│   │   Signal                                                 │   │
│   │   Strength                                               │   │
│   │       ^                                                  │   │
│   │       │     ╱╲                                          │   │
│   │       │    ╱  ╲                                         │   │
│   │       │   ╱    ╲                                        │   │
│   │       │  ╱      ╲                                       │   │
│   │       │ ╱        ╲                                      │   │
│   │       └──┬────────┬──► Frequency                        │   │
│   │          f1      f2                                     │   │
│   │          ◄────────►                                     │   │
│   │           Bandwidth = f2 - f1 (Hz)                      │   │
│   │                                                          │   │
│   │   Example: Phone voice bandwidth = 3,400 - 300 = 3,100 Hz│  │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. Digital Bandwidth (bps)                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Amount of data transmitted per unit time              │   │
│   │                                                          │   │
│   │   ┌───────────────────────────────────────┐             │   │
│   │   │     Channel Capacity (Bandwidth)       │             │   │
│   │   │     ══════════════════════════════    │             │   │
│   │   │         100 Mbps                       │             │   │
│   │   └───────────────────────────────────────┘             │   │
│   │                                                          │   │
│   │   Units:                                                 │   │
│   │   - bps (bits per second)                               │   │
│   │   - Kbps = 1,000 bps                                    │   │
│   │   - Mbps = 1,000,000 bps                                │   │
│   │   - Gbps = 1,000,000,000 bps                            │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Throughput and Latency

```
┌─────────────────────────────────────────────────────────────────┐
│                    Throughput & Latency                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Bandwidth vs Throughput:                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Bandwidth:                                             │   │
│   │   - Theoretical maximum transmission capacity           │   │
│   │   - Pipe diameter                                       │   │
│   │                                                          │   │
│   │   Throughput:                                            │   │
│   │   - Actual amount of data transmitted                   │   │
│   │   - Always ≤ bandwidth                                  │   │
│   │   - Throughput ≤ Bandwidth                              │   │
│   │                                                          │   │
│   │   ┌───────────────────────────────────────┐             │   │
│   │   │  Bandwidth: 100 Mbps                   │             │   │
│   │   │  ═══════════════════════════════════  │             │   │
│   │   │  Actual Throughput: 80 Mbps            │             │   │
│   │   │  ════════════════════════             │             │   │
│   │   │  (Reduced by overhead, congestion, etc)│             │   │
│   │   └───────────────────────────────────────┘             │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Latency:                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Time for data to travel from source to destination    │   │
│   │                                                          │   │
│   │   Total Latency = Propagation + Transmission + Queuing  │   │
│   │                   + Processing Delay                     │   │
│   │                                                          │   │
│   │   1. Propagation Delay                                  │   │
│   │      - Time for signal to travel through medium         │   │
│   │      - Distance / Propagation speed                     │   │
│   │      - Speed of light ≈ 3×10^8 m/s (vacuum)            │   │
│   │                                                          │   │
│   │   2. Transmission Delay                                 │   │
│   │      - Time to put data on link                         │   │
│   │      - Data size / Bandwidth                            │   │
│   │                                                          │   │
│   │   3. Queuing Delay                                      │   │
│   │      - Waiting time in router                           │   │
│   │      - Varies with network congestion                   │   │
│   │                                                          │   │
│   │   4. Processing Delay                                   │   │
│   │      - Header check, routing decision time              │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   RTT (Round Trip Time):                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Time for packet to make round trip                    │   │
│   │                                                          │   │
│   │   Client ────────────────────► Server                   │   │
│   │            ◄────────────────────                        │   │
│   │            ◄──────────────────────►                     │   │
│   │                      RTT                                │   │
│   │                                                          │   │
│   │   Measurement: ping command                             │   │
│   │   $ ping google.com                                     │   │
│   │   time=15.2 ms  ← This is RTT                          │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Shannon Capacity Formula

```
┌─────────────────────────────────────────────────────────────────┐
│                      Channel Capacity Formulas                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Nyquist Formula (Noiseless channel):                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   C = 2 × B × log₂(L)                                    │   │
│   │                                                          │   │
│   │   C: Channel capacity (bps)                             │   │
│   │   B: Bandwidth (Hz)                                     │   │
│   │   L: Number of signal levels                            │   │
│   │                                                          │   │
│   │   Example: Bandwidth 3,100 Hz, 2 signal levels          │   │
│   │   C = 2 × 3,100 × log₂(2) = 6,200 bps                   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Shannon Formula (Noisy channel):                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   C = B × log₂(1 + S/N)                                  │   │
│   │                                                          │   │
│   │   C: Channel capacity (bps)                             │   │
│   │   B: Bandwidth (Hz)                                     │   │
│   │   S/N: Signal-to-Noise Ratio                            │   │
│   │                                                          │   │
│   │   SNR(dB) = 10 × log₁₀(S/N)                             │   │
│   │                                                          │   │
│   │   Example: Bandwidth 1 MHz, SNR 63 (18 dB)              │   │
│   │   C = 1,000,000 × log₂(1 + 63)                          │   │
│   │   C = 1,000,000 × 6 = 6 Mbps                            │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Ethernet Cable Types

### UTP Cable Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                    UTP Cable Categories                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┬──────────┬──────────┬────────┬─────────────────┐  │
│   │Category │  Speed   │Bandwidth │Distance│     Usage       │  │
│   ├─────────┼──────────┼──────────┼────────┼─────────────────┤  │
│   │  Cat3   │ 10 Mbps  │  16 MHz  │ 100m   │ Phone, legacy   │  │
│   ├─────────┼──────────┼──────────┼────────┼─────────────────┤  │
│   │  Cat5   │ 100 Mbps │ 100 MHz  │ 100m   │ 100BASE-TX      │  │
│   │         │          │          │        │ (Rarely used)   │  │
│   ├─────────┼──────────┼──────────┼────────┼─────────────────┤  │
│   │  Cat5e  │ 1 Gbps   │ 100 MHz  │ 100m   │ 1000BASE-T      │  │
│   │         │          │          │        │ (Current std)   │  │
│   ├─────────┼──────────┼──────────┼────────┼─────────────────┤  │
│   │  Cat6   │ 10 Gbps  │ 250 MHz  │ 55m    │ 10GBASE-T       │  │
│   │         │ 1 Gbps   │          │ 100m   │ (10G up to 55m) │  │
│   ├─────────┼──────────┼──────────┼────────┼─────────────────┤  │
│   │  Cat6a  │ 10 Gbps  │ 500 MHz  │ 100m   │ 10GBASE-T       │  │
│   │         │          │          │        │ (Data centers)  │  │
│   ├─────────┼──────────┼──────────┼────────┼─────────────────┤  │
│   │  Cat7   │ 10 Gbps  │ 600 MHz  │ 100m   │ STP, shielded   │  │
│   │         │          │          │        │ (Special conn.) │  │
│   ├─────────┼──────────┼──────────┼────────┼─────────────────┤  │
│   │  Cat8   │ 25/40    │ 2000 MHz │ 30m    │ Data centers    │  │
│   │         │ Gbps     │          │        │ (Server links)  │  │
│   └─────────┴──────────┴──────────┴────────┴─────────────────┘  │
│                                                                  │
│   Recommendations:                                              │
│   - Home/Office: Cat5e or Cat6                                  │
│   - High-speed networks: Cat6a                                  │
│   - Data centers: Cat6a, Cat7, Cat8                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Cable Wiring Standards

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cable Wiring Standards                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   T568A vs T568B Wiring:                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   T568A:                        T568B:                   │   │
│   │   ┌───────────────────┐        ┌───────────────────┐    │   │
│   │   │ 1: W-Green ██     │        │ 1: W-Orange ██    │    │   │
│   │   │ 2: Green   ██     │        │ 2: Orange   ██    │    │   │
│   │   │ 3: W-Orange ██    │        │ 3: W-Green  ██    │    │   │
│   │   │ 4: Blue    ██     │        │ 4: Blue     ██    │    │   │
│   │   │ 5: W-Blue  ██     │        │ 5: W-Blue   ██    │    │   │
│   │   │ 6: Orange  ██     │        │ 6: Green    ██    │    │   │
│   │   │ 7: W-Brown ██     │        │ 7: W-Brown  ██    │    │   │
│   │   │ 8: Brown   ██     │        │ 8: Brown    ██    │    │   │
│   │   └───────────────────┘        └───────────────────┘    │   │
│   │                                                          │   │
│   │   * T568B is more commonly used                          │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Cable Types:                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   1. Straight-Through Cable                             │   │
│   │      - Both ends use same standard (T568B-T568B)        │   │
│   │      - PC ↔ Switch, Switch ↔ Router                    │   │
│   │      - Most common                                      │   │
│   │                                                          │   │
│   │      [PC] ═══════════════════════════════ [Switch]      │   │
│   │            1-1, 2-2, 3-3, 4-4...                        │   │
│   │                                                          │   │
│   │   2. Crossover Cable                                    │   │
│   │      - Different standards at ends (T568A-T568B)        │   │
│   │      - TX and RX pins crossed                           │   │
│   │      - PC ↔ PC, Switch ↔ Switch                        │   │
│   │                                                          │   │
│   │      [PC1] ════════╳════════════════════ [PC2]          │   │
│   │             1-3, 2-6 crossed                            │   │
│   │                                                          │   │
│   │   3. Rollover Cable (Console Cable)                     │   │
│   │      - All pins reversed (1-8, 2-7...)                  │   │
│   │      - PC ↔ Router/Switch console port                 │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   * Modern equipment supports Auto-MDIX for automatic detection │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pin Layout and Usage

```
┌─────────────────────────────────────────────────────────────────┐
│                      Ethernet Pin Layout                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   10BASE-T / 100BASE-TX (4 pins used):                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   RJ-45 Connector Pin Layout:                           │   │
│   │                                                          │   │
│   │   ┌─────────────────────────┐                           │   │
│   │   │  1  2  3  4  5  6  7  8 │  ← Pin number            │   │
│   │   │  ▓  ▓  ▓  □  □  ▓  □  □ │                           │   │
│   │   │ TX+ TX- RX+       RX-   │                           │   │
│   │   └─────────────────────────┘                           │   │
│   │                                                          │   │
│   │   Pin 1, 2: Transmit (TX+, TX-)                         │   │
│   │   Pin 3, 6: Receive (RX+, RX-)                          │   │
│   │   Pin 4, 5, 7, 8: Unused (can be used for PoE)          │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   1000BASE-T / Gigabit (All 8 pins used):                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌─────────────────────────┐                           │   │
│   │   │  1  2  3  4  5  6  7  8 │                           │   │
│   │   │  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓ │  ← All pins used         │   │
│   │   │ BI-DA  BI-DB  BI-DC  BI-DD │                        │   │
│   │   └─────────────────────────┘                           │   │
│   │                                                          │   │
│   │   - All 4 pairs used bidirectionally (Full-Duplex)      │   │
│   │   - Uses 5-level PAM (PAM-5) signaling                  │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Connector Types

### Coaxial/UTP Connectors

```
┌─────────────────────────────────────────────────────────────────┐
│                      Network Connectors                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   RJ-45 (8P8C):                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Most common Ethernet connector                        │   │
│   │                                                          │   │
│   │      ┌─────────────┐                                    │   │
│   │      │  ┌───────┐  │                                    │   │
│   │      │  │▓▓▓▓▓▓▓│  │  ← 8 contacts                     │   │
│   │      │  └───────┘  │                                    │   │
│   │      │      ▓      │  ← Latch clip                     │   │
│   │      │     ═══     │                                    │   │
│   │      │    ═════    │                                    │   │
│   │      │   ═══════   │  ← Cable                          │   │
│   │      └─────────────┘                                    │   │
│   │                                                          │   │
│   │   Usage: Ethernet (UTP cable)                           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   RJ-11 (6P2C/6P4C):                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Phone line connector (smaller than RJ-45)             │   │
│   │                                                          │   │
│   │      ┌─────────┐                                        │   │
│   │      │ ┌─────┐ │  ← 4 or 2 contacts                    │   │
│   │      │ └─────┘ │                                        │   │
│   │      │    ▓    │                                        │   │
│   │      │  ═════  │                                        │   │
│   │      └─────────┘                                        │   │
│   │                                                          │   │
│   │   Usage: Phone, modem, DSL                              │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   BNC (Bayonet Neill-Concelman):                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Coaxial cable connector                               │   │
│   │                                                          │   │
│   │      ┌─────┐                                            │   │
│   │      │     │─────────  ← Center pin                     │   │
│   │      │  ○  │                                            │   │
│   │      │     │                                            │   │
│   │      └──┬──┘  ← Locking ring (twist to lock)           │   │
│   │         │                                                │   │
│   │                                                          │   │
│   │   Usage: Legacy LAN (10BASE2), CCTV                     │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Fiber Optic Connectors

```
┌─────────────────────────────────────────────────────────────────┐
│                     Fiber Optic Connectors                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   SC (Subscriber Connector):                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │      ┌─────────┐                                        │   │
│   │      │    □    │  ← Square push-pull type              │   │
│   │      │    ○    │  ← 2.5mm ferrule                      │   │
│   │      │    □    │                                        │   │
│   │      └────┬────┘                                        │   │
│   │           │                                              │   │
│   │                                                          │   │
│   │   - Most common fiber connector                         │   │
│   │   - Used in data centers, FTTH                          │   │
│   │   - Simple push-pull connection                         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   LC (Lucent Connector):                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │      ┌───────┐                                          │   │
│   │      │   ○   │  ← Half size of SC                       │   │
│   │      │   ▓   │  ← 1.25mm ferrule                        │   │
│   │      └───┬───┘  ← Latch clip                           │   │
│   │          │                                               │   │
│   │                                                          │   │
│   │   - 50% smaller than SC                                 │   │
│   │   - Suitable for high-density connections               │   │
│   │   - Used with SFP transceivers                          │   │
│   │   - Currently most popular fiber connector              │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   SFP (Small Form-factor Pluggable):                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌───────────────────────────┐                         │   │
│   │   │  ┌─────┐      ┌─────┐   │                         │   │
│   │   │  │  TX │      │  RX │   │  ← TX/RX separate       │   │
│   │   │  └─────┘      └─────┘   │                         │   │
│   │   │         SFP              │                         │   │
│   │   └───────────────────────────┘                         │   │
│   │                                                          │   │
│   │   - Hot-swappable transceiver                           │   │
│   │   - Supports various speeds/distances                   │   │
│   │   - SFP (1G), SFP+ (10G), SFP28 (25G), QSFP+ (40G)      │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ST (Straight Tip):                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │      ○────────  ← Bayonet (twist to lock) style         │   │
│   │                                                          │   │
│   │   - Legacy fiber connector                              │   │
│   │   - Used in multi-mode networks                         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Connector Comparison:                                         │
│   ┌─────────┬───────────┬───────────────┬───────────────────┐   │
│   │Connector│Ferrule    │Connection Type│      Usage        │   │
│   ├─────────┼───────────┼───────────────┼───────────────────┤   │
│   │   SC    │  2.5mm    │   Push-pull   │ General, FTTH     │   │
│   │   LC    │  1.25mm   │   Latch       │ Data center, SFP  │   │
│   │   ST    │  2.5mm    │   Bayonet     │ Legacy systems    │   │
│   │  FC     │  2.5mm    │   Screw       │ Precision meas.   │   │
│   │  MPO    │  Multiple │   Push-pull   │ High density (40G)│   │
│   └─────────┴───────────┴───────────────┴───────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Wireless Transmission

### Wireless Communication Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Wireless Communication Overview                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Radio Frequency Spectrum:                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Frequency   │        Usage                             │   │
│   │   ───────────────────────────────────────────           │   │
│   │                                                          │   │
│   │   3 kHz - 300 kHz    : Low freq (AM radio)              │   │
│   │   300 kHz - 3 MHz    : Medium wave (AM radio)           │   │
│   │   3 MHz - 30 MHz     : Short wave (Radio, amateur)      │   │
│   │   30 MHz - 300 MHz   : VHF (FM radio, TV)               │   │
│   │   300 MHz - 3 GHz    : UHF (TV, cellular, Wi-Fi)        │   │
│   │   3 GHz - 30 GHz     : Microwave (satellite, 5G)        │   │
│   │   30 GHz - 300 GHz   : Millimeter wave (5G, radar)      │   │
│   │                                                          │   │
│   │   ◄─ Lower frequency ─────────── Higher frequency ─►    │   │
│   │   ◄─ Long range, low speed ──── Short range, high speed─►│  │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Wi-Fi (IEEE 802.11)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wi-Fi (IEEE 802.11)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Wi-Fi Standard Evolution:                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌──────────┬────────┬─────────┬─────────┬───────────┐ │   │
│   │   │ Standard │  Freq  │Max Speed│  Range  │ Marketing │ │   │
│   │   ├──────────┼────────┼─────────┼─────────┼───────────┤ │   │
│   │   │  802.11b │ 2.4GHz │  11Mbps │  ~35m   │     -     │ │   │
│   │   │  802.11a │  5GHz  │  54Mbps │  ~25m   │     -     │ │   │
│   │   │  802.11g │ 2.4GHz │  54Mbps │  ~35m   │     -     │ │   │
│   │   │  802.11n │ 2.4/5  │ 600Mbps │  ~50m   │   Wi-Fi 4 │ │   │
│   │   │ 802.11ac │  5GHz  │ 6.9Gbps │  ~35m   │   Wi-Fi 5 │ │   │
│   │   │ 802.11ax │ 2.4/5/6│ 9.6Gbps │  ~35m   │   Wi-Fi 6 │ │   │
│   │   │802.11ax-6│  6GHz  │ 9.6Gbps │  ~35m   │  Wi-Fi 6E │ │   │
│   │   │ 802.11be │2.4/5/6 │ 46Gbps  │  ~35m   │   Wi-Fi 7 │ │   │
│   │   └──────────┴────────┴─────────┴─────────┴───────────┘ │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2.4 GHz vs 5 GHz:                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   2.4 GHz Band:                                         │   │
│   │   ┌─────────────────────────────────────────────┐       │   │
│   │   │ + Long range, good wall penetration          │       │   │
│   │   │ + Legacy device compatibility                │       │   │
│   │   │ - Congested (Bluetooth, microwave interference)│     │   │
│   │   │ - Few channels (3 non-overlapping channels)  │       │   │
│   │   └─────────────────────────────────────────────┘       │   │
│   │                                                          │   │
│   │   5 GHz Band:                                           │   │
│   │   ┌─────────────────────────────────────────────┐       │   │
│   │   │ + More channels (23 non-overlapping)         │       │   │
│   │   │ + Less congested, less interference          │       │   │
│   │   │ + Higher speeds                              │       │   │
│   │   │ - Shorter range, poor wall penetration       │       │   │
│   │   └─────────────────────────────────────────────┘       │   │
│   │                                                          │   │
│   │   6 GHz Band (Wi-Fi 6E):                                │   │
│   │   ┌─────────────────────────────────────────────┐       │   │
│   │   │ + Least interference                         │       │   │
│   │   │ + Very wide bandwidth                        │       │   │
│   │   │ - Only newest devices support                │       │   │
│   │   │ - Shortest range                             │       │   │
│   │   └─────────────────────────────────────────────┘       │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Wi-Fi Security:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   WEP (Wired Equivalent Privacy):                       │   │
│   │   - Legacy, vulnerable (do not use)                     │   │
│   │                                                          │   │
│   │   WPA (Wi-Fi Protected Access):                         │   │
│   │   - TKIP encryption, better than WEP                    │   │
│   │                                                          │   │
│   │   WPA2:                                                 │   │
│   │   - AES encryption, current standard                    │   │
│   │   - Personal (PSK), Enterprise (RADIUS)                 │   │
│   │                                                          │   │
│   │   WPA3:                                                 │   │
│   │   - Latest standard (2018)                              │   │
│   │   - SAE (Simultaneous Authentication), improved security│   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Other Wireless Technologies

```
┌─────────────────────────────────────────────────────────────────┐
│                     Other Wireless Technologies                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Bluetooth:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   - Frequency: 2.4 GHz ISM band                         │   │
│   │   - Range: ~10m (Class 2), ~100m (Class 1)              │   │
│   │   │   - Speed: ~3 Mbps (Classic), ~2 Mbps (BLE)              │   │
│   │   - Usage: Earphones, keyboards, IoT sensors            │   │
│   │   - Versions: 4.0 (BLE), 5.0 (long range), 5.2 (LE Audio)│  │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Zigbee (IEEE 802.15.4):                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   - Frequency: 2.4 GHz                                  │   │
│   │   - Speed: 250 Kbps                                     │   │
│   │   - Low power, mesh network                             │   │
│   │   - Usage: Smart home, industrial automation            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Cellular (3G/4G/5G):                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌────────┬───────────┬───────────────────────────────┐│   │
│   │   │  Gen   │  Speed    │        Features               ││   │
│   │   ├────────┼───────────┼───────────────────────────────┤│   │
│   │   │   3G   │ ~2 Mbps   │ Video calls, mobile internet  ││   │
│   │   │   4G   │ ~100 Mbps │ High-speed data, VoLTE        ││   │
│   │   │  LTE   │ ~1 Gbps   │ (4G enhanced)                 ││   │
│   │   │   5G   │ ~20 Gbps  │ Ultra-low latency, mMTC, eMBB ││   │
│   │   └────────┴───────────┴───────────────────────────────┘│   │
│   │                                                          │   │
│   │   5G Features:                                          │   │
│   │   - eMBB: Enhanced Mobile Broadband                     │   │
│   │   - URLLC: Ultra-Reliable Low-Latency Communication     │   │
│   │   - mMTC: Massive IoT connectivity                      │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Basic Problems

**1. Explain three major functions of the physical layer.**

**2. Arrange the following media in order of transmission distance (longest first):**
   - UTP, Single-mode fiber, Coaxial cable, Multi-mode fiber

**3. Explain the differences between Cat5e and Cat6 cables.**

**4. Explain the usage of straight-through and crossover cables.**

### Application Problems

**5. Choose appropriate transmission media for the following situations and explain why:**
   - (a) Connecting 100 PCs in an office
   - (b) Connecting two buildings 1km apart
   - (c) 40Gbps connection between servers in data center

**6. Compare the advantages and disadvantages of 2.4GHz and 5GHz Wi-Fi.**

**7. Calculate transmission delay and propagation delay when transmitting 1Gbps data through a 100m Ethernet cable. (Assume speed of light in cable is 2x10^8 m/s)**

### Advanced Problems

**8. Using Shannon's formula, calculate the maximum transmission speed of a channel with 10MHz bandwidth and SNR of 1000.**

**9. Explain at least 5 advantages of fiber optic over copper cable.**

**10. Explain why fiber optic is chosen over Cat6a in data centers.**

---

<details>
<summary>Answers</summary>

**1.**
- Bit synchronization (TX/RX clock sync)
- Bit rate control (bps determination)
- Transmission mode definition (simplex/half-duplex/full-duplex)
- Physical interface definition (connectors, voltage)

**2.** Single-mode fiber (~100km) > Multi-mode fiber (~2km) > Coaxial cable (~500m) > UTP (100m)

**3.**
- Cat5e: Max 1Gbps, 100MHz bandwidth
- Cat6: Max 10Gbps (55m), 250MHz bandwidth
- Cat6 provides higher bandwidth and better interference immunity

**4.**
- Straight-through: PC-Switch, Switch-Router (different device types)
- Crossover: PC-PC, Switch-Switch (same device types)

**5.**
- (a) Cat5e or Cat6 UTP - Inexpensive and suitable for under 100m
- (b) Single-mode fiber - Long distance, EMI immune
- (c) Multi-mode fiber (OM4) or Single-mode - High speed, data center standard

**6.**
- 2.4GHz: Wide range, good wall penetration, congested, lower speed
- 5GHz: Narrow range, poor wall penetration, less congested, higher speed

**7.**
- Transmission delay = Data size / Bandwidth
  = (e.g., 1500 bytes) / 1Gbps = 12 microseconds
- Propagation delay = Distance / Speed = 100m / (2x10^8 m/s) = 0.5 microseconds

**8.**
C = B × log2(1 + S/N)
= 10,000,000 × log2(1001)
≈ 10,000,000 × 9.97
≈ 99.7 Mbps

**9.**
1. Complete EMI immunity
2. Very high bandwidth (terabit-class)
3. Long distance capable
4. Difficult to tap (security)
5. Light weight
6. No corrosion
7. Low signal attenuation

**10.**
- Need for 40G/100G+ speeds
- EMI interference environment
- Future scalability
- Long distance needed (between cabinets)
- Reduced cable density

</details>

---

**Previous**: [TCP/IP Model](./03_TCP_IP_Model.md) | **Next**: [Data Link Layer](./05_Data_Link_Layer.md)

---

## References

- Data Communications and Networking (Forouzan)
- [ANSI/TIA-568 Cabling Standards](https://www.tiaonline.org/)
- [IEEE 802.3 Ethernet Standard](https://www.ieee802.org/3/)
- [Wi-Fi Alliance](https://www.wi-fi.org/)
