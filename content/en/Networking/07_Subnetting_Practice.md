# Subnetting Practice

**Previous**: [IP Addressing and Subnetting](./06_IP_Address_Subnetting.md) | **Next**: [Routing Basics](./08_Routing_Basics.md)

**Difficulty**: ⭐⭐⭐
**Estimated Learning Time**: 3-4 hours
**Prerequisites**: [06_IP_Address_Subnetting.md](./06_IP_Address_Subnetting.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Calculate network address, broadcast address, and usable host range for any given CIDR block
2. Determine the required subnet mask given a number of hosts or subnets
3. Perform subnet division to split a network into equal-sized subnets
4. Apply VLSM (Variable Length Subnet Masking) to allocate subnets of different sizes efficiently
5. Design a subnetting plan for a multi-site network scenario
6. Verify subnetting calculations using binary arithmetic and powers of two

---

Subnetting is where IP addressing theory meets hands-on practice. Network engineers perform these calculations daily -- whether provisioning a new office, sizing a data center VLAN, or optimizing address allocation across WAN links. The ability to quickly and accurately subnet is a core competency tested in every networking certification and job interview.

## Table of Contents

1. [Subnetting Calculation Basics](#1-subnetting-calculation-basics)
2. [Network Address, Broadcast Address, and Host Range](#2-network-address-broadcast-address-and-host-range)
3. [Subnet Division Examples](#3-subnet-division-examples)
4. [VLSM (Variable Length Subnet Mask)](#4-vlsm-variable-length-subnet-mask)
5. [Subnet Design Problems](#5-subnet-design-problems)
6. [Practice Problems](#6-practice-problems)
7. [Next Steps](#7-next-steps)
8. [References](#8-references)

---

## 1. Subnetting Calculation Basics

### 1.1 Memorizing Powers of 2

To calculate subnetting quickly, you must memorize powers of 2.

```
2^0 = 1        2^5 = 32       2^10 = 1,024
2^1 = 2        2^6 = 64       2^11 = 2,048
2^2 = 4        2^7 = 128      2^12 = 4,096
2^3 = 8        2^8 = 256      2^13 = 8,192
2^4 = 16       2^9 = 512      2^14 = 16,384
```

### 1.2 Subnet Mask and CIDR Notation

| CIDR | Subnet Mask      | Host Bits | Usable Hosts |
|------|------------------|-----------|--------------|
| /24  | 255.255.255.0    | 8         | 254          |
| /25  | 255.255.255.128  | 7         | 126          |
| /26  | 255.255.255.192  | 6         | 62           |
| /27  | 255.255.255.224  | 5         | 30           |
| /28  | 255.255.255.240  | 4         | 14           |
| /29  | 255.255.255.248  | 3         | 6            |
| /30  | 255.255.255.252  | 2         | 2            |
| /31  | 255.255.255.254  | 1         | 2 (Point-to-Point) |
| /32  | 255.255.255.255  | 0         | 1 (Host route) |

### 1.3 Magic Number

The magic number represents the subnet size and speeds up calculations.

```
Magic Number = 256 - Last octet value of subnet mask
```

| Subnet Mask     | Magic Number | CIDR |
|-----------------|--------------|------|
| 255.255.255.0   | 256          | /24  |
| 255.255.255.128 | 128          | /25  |
| 255.255.255.192 | 64           | /26  |
| 255.255.255.224 | 32           | /27  |
| 255.255.255.240 | 16           | /28  |
| 255.255.255.248 | 8            | /29  |
| 255.255.255.252 | 4            | /30  |

### 1.4 Subnet Calculation Formulas

```
Subnet Count = 2^(subnet bits)
Host Count = 2^(host bits) - 2
Block Size = 256 - Subnet mask value (last octet)
```

---

## 2. Network Address, Broadcast Address, and Host Range

### 2.1 Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                      192.168.1.0/24                         │
├─────────────────────────────────────────────────────────────┤
│  Network Address   : 192.168.1.0     (all host bits = 0)   │
│  First Host        : 192.168.1.1                            │
│  Last Host         : 192.168.1.254                          │
│  Broadcast Address : 192.168.1.255   (all host bits = 1)   │
│  Usable Hosts      : 254                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Calculation Method

**Example: Calculate network information for 192.168.1.100/26**

**Step 1: Determine subnet mask**
```
/26 = 255.255.255.192
Binary: 11111111.11111111.11111111.11000000
```

**Step 2: Calculate block size**
```
Block size = 256 - 192 = 64
```

**Step 3: Find network address**
```
Last octet of IP: 100
100 ÷ 64 = 1 (remainder 36)
Network start: 1 × 64 = 64
Network address: 192.168.1.64
```

**Step 4: Calculate broadcast address**
```
Broadcast = Network address + Block size - 1
          = 64 + 64 - 1 = 127
Broadcast address: 192.168.1.127
```

**Step 5: Determine host range**
```
First host: 192.168.1.65
Last host: 192.168.1.126
Usable hosts: 62 (2^6 - 2)
```

### 2.3 Visual Understanding

```
Subnet division of 192.168.1.0/26 network

┌─────────────────────────────────────────────────────────┐
│                    192.168.1.0/24                       │
├─────────────────────────────────────────────────────────┤
│ Subnet 1       │ Subnet 2       │ Subnet 3       │ Subnet 4       │
│ .0 - .63      │ .64 - .127    │ .128 - .191   │ .192 - .255   │
│ /26           │ /26           │ /26           │ /26           │
│               │               │               │               │
│ Network: .0   │ Network: .64  │ Network: .128 │ Network: .192 │
│ Hosts:        │ Hosts:        │ Hosts:        │ Hosts:        │
│ .1 - .62      │ .65 - .126    │ .129 - .190   │ .193 - .254   │
│ Broadcast:.63 │ Broadcast:.127│ Broadcast:.191│ Broadcast:.255│
└─────────────────────────────────────────────────────────┘
```

### 2.4 Binary Calculation

```
IP Address:    192.168.1.100 = 11000000.10101000.00000001.01100100
Subnet Mask:   255.255.255.192 = 11111111.11111111.11111111.11000000
                                                          ↑↑
                                                    Network bits

AND operation (Network address):
  11000000.10101000.00000001.01100100  (IP)
& 11111111.11111111.11111111.11000000  (Mask)
= 11000000.10101000.00000001.01000000  = 192.168.1.64

OR operation (Broadcast address):
  11000000.10101000.00000001.01000000  (Network)
| 00000000.00000000.00000000.00111111  (Wildcard)
= 11000000.10101000.00000001.01111111  = 192.168.1.127
```

---

## 3. Subnet Division Examples

### 3.1 Dividing /24 Network into 4 Parts

**Problem**: Divide 10.0.0.0/24 into 4 equal subnets

**Solution**:
```
4 subnets = 2^2 → Need to add 2 subnet bits
New CIDR: /24 + 2 = /26
Hosts per subnet: 2^6 - 2 = 62
```

**Results**:

| Subnet | Network Address | Host Range       | Broadcast    |
|--------|----------------|------------------|--------------|
| 1      | 10.0.0.0/26    | 10.0.0.1 - 62    | 10.0.0.63    |
| 2      | 10.0.0.64/26   | 10.0.0.65 - 126  | 10.0.0.127   |
| 3      | 10.0.0.128/26  | 10.0.0.129 - 190 | 10.0.0.191   |
| 4      | 10.0.0.192/26  | 10.0.0.193 - 254 | 10.0.0.255   |

### 3.2 Dividing /16 Network into 16 Parts

**Problem**: Divide 172.16.0.0/16 into 16 subnets

**Solution**:
```
16 subnets = 2^4 → Add 4 subnet bits
New CIDR: /16 + 4 = /20
Hosts per subnet: 2^12 - 2 = 4,094
Block size: 256 - 240 = 16 (in third octet)
```

**Results**:

| Subnet | Network Address    | Host Range                     |
|--------|--------------------|-------------------------------|
| 1      | 172.16.0.0/20      | 172.16.0.1 - 172.16.15.254    |
| 2      | 172.16.16.0/20     | 172.16.16.1 - 172.16.31.254   |
| 3      | 172.16.32.0/20     | 172.16.32.1 - 172.16.47.254   |
| 4      | 172.16.48.0/20     | 172.16.48.1 - 172.16.63.254   |
| ...    | ...                | ...                            |
| 16     | 172.16.240.0/20    | 172.16.240.1 - 172.16.255.254 |

### 3.3 Subnet for Specific Host Count

**Problem**: Need subnet to support 50 hosts from 192.168.10.0/24

**Solution**:
```
Need to support 50 hosts
2^5 - 2 = 30 (insufficient)
2^6 - 2 = 62 (sufficient)

Host bits: 6
CIDR: /26
Subnet mask: 255.255.255.192
```

---

## 4. VLSM (Variable Length Subnet Mask)

### 4.1 What is VLSM?

VLSM is a technique to efficiently allocate IP addresses using subnets of different sizes.

```
Traditional Subnetting (Equal Size)     VLSM (Variable Size)

┌─────────┬─────────┐           ┌─────────────────┐
│ /26     │ /26     │           │     /25         │
│ 62 host │ 62 host │           │   126 host      │
├─────────┼─────────┤           ├────────┬────────┤
│ /26     │ /26     │           │ /27    │ /27    │
│ 62 host │ 62 host │           │30 host │30 host │
└─────────┴─────────┘           ├───┬────┴────────┤
                                │/30│   /28       │
Wastes many IPs                 │2  │   14 host   │
                                └───┴─────────────┘
                                Efficient IP usage
```

### 4.2 VLSM Design Principles

1. **Allocate large subnets first**: Start with networks requiring most hosts
2. **Use powers of 2**: Each subnet size is 2^n - 2
3. **Use contiguous addresses**: Maintain contiguous address space

### 4.3 VLSM Design Example

**Scenario**: Divide 172.20.0.0/22 according to requirements

| Department    | Required Hosts |
|--------------|----------------|
| Sales        | 200            |
| Development  | 100            |
| HR           | 50             |
| Management   | 20             |
| WAN Link 1   | 2              |
| WAN Link 2   | 2              |

**Step 1: Sort by size and determine subnets**

| Department   | Hosts | Required Bits | Subnet | Actual Hosts |
|-------------|-------|--------------|--------|--------------|
| Sales       | 200   | 8            | /24    | 254          |
| Development | 100   | 7            | /25    | 126          |
| HR          | 50    | 6            | /26    | 62           |
| Management  | 20    | 5            | /27    | 30           |
| WAN Link 1  | 2     | 2            | /30    | 2            |
| WAN Link 2  | 2     | 2            | /30    | 2            |

**Step 2: Address allocation**

```
172.20.0.0/22 (1,022 usable hosts)
│
├─ Sales: 172.20.0.0/24
│   ├─ Network: 172.20.0.0
│   ├─ Hosts: 172.20.0.1 - 172.20.0.254
│   └─ Broadcast: 172.20.0.255
│
├─ Development: 172.20.1.0/25
│   ├─ Network: 172.20.1.0
│   ├─ Hosts: 172.20.1.1 - 172.20.1.126
│   └─ Broadcast: 172.20.1.127
│
├─ HR: 172.20.1.128/26
│   ├─ Network: 172.20.1.128
│   ├─ Hosts: 172.20.1.129 - 172.20.1.190
│   └─ Broadcast: 172.20.1.191
│
├─ Management: 172.20.1.192/27
│   ├─ Network: 172.20.1.192
│   ├─ Hosts: 172.20.1.193 - 172.20.1.222
│   └─ Broadcast: 172.20.1.223
│
├─ WAN Link 1: 172.20.1.224/30
│   ├─ Network: 172.20.1.224
│   ├─ Hosts: 172.20.1.225 - 172.20.1.226
│   └─ Broadcast: 172.20.1.227
│
└─ WAN Link 2: 172.20.1.228/30
    ├─ Network: 172.20.1.228
    ├─ Hosts: 172.20.1.229 - 172.20.1.230
    └─ Broadcast: 172.20.1.231
```

**Step 3: Address usage status**

```
172.20.0.0/22 Address Map

       .0         .64        .128       .192       .255
        ├──────────┼──────────┼──────────┼──────────┤
   .0   │                  Sales /24                  │
        ├─────────────────────┼──────────┼────┬─────┤
   .1   │    Development /25  │   HR     │Mgmt│ WAN │
        │                     │  /26     │/27 │/30  │
        └─────────────────────┴──────────┴────┴─────┘
   .2   │              (Reserved)                    │
        ├────────────────────────────────────────────┤
   .3   │              (Reserved)                    │
        └────────────────────────────────────────────┘
```

### 4.4 VLSM Advantages and Disadvantages

**Advantages**:
- Efficient IP address usage
- Flexible allocation matching network size
- Route aggregation possible

**Disadvantages**:
- Increased design complexity
- Address overlap possible if mistakes made
- All routers must support VLSM

---

## 5. Subnet Design Problems

### 5.1 Small Office Design

**Requirements**:
- Public IP: 203.0.113.0/28
- Networks: Web servers (3), Internal servers (5), Management (2)

**Design**:
```
203.0.113.0/28 (14 usable hosts)
│
├─ Web servers: 203.0.113.1 - 203.0.113.3
├─ Internal servers: 203.0.113.4 - 203.0.113.8
├─ Management: 203.0.113.9 - 203.0.113.10
├─ Reserved: 203.0.113.11 - 203.0.113.13
└─ Gateway: 203.0.113.14 (last host)
```

### 5.2 Medium Enterprise Network

**Requirements**:
- Network: 10.100.0.0/16
- Headquarters: 1,000 people
- Branch A: 200 people
- Branch B: 150 people
- Data center: 500 servers
- Consider future expansion

**VLSM Design**:

| Zone          | Subnet         | Host Range                     | Capacity |
|--------------|----------------|--------------------------------|----------|
| Headquarters | 10.100.0.0/22  | 10.100.0.1 - 10.100.3.254      | 1,022    |
| Data Center  | 10.100.4.0/23  | 10.100.4.1 - 10.100.5.254      | 510      |
| Branch A     | 10.100.6.0/24  | 10.100.6.1 - 10.100.6.254      | 254      |
| Branch B     | 10.100.7.0/24  | 10.100.7.1 - 10.100.7.254      | 254      |
| Mgmt VLAN    | 10.100.8.0/26  | 10.100.8.1 - 10.100.8.62       | 62       |
| WAN Link     | 10.100.8.64/30 | 10.100.8.65 - 10.100.8.66      | 2        |
| Reserved     | 10.100.9.0/24+ | -                              | -        |

### 5.3 Network Diagram

```
                    ┌─────────────┐
                    │   Internet  │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  Edge Router │
                    │ 10.100.8.65 │
                    └──────┬──────┘
              WAN Link /30 │ 10.100.8.64/30
                    ┌──────┴──────┐
                    │ Core Switch │
                    │ 10.100.8.1  │
                    └──────┬──────┘
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────┴──────┐┌──────┴──────┐┌──────┴──────┐
     │Headquarters ││ Data Center ││   Branch A  │
     │10.100.0.0/22││10.100.4.0/23││10.100.6.0/24│
     │ 1,022 hosts ││  510 hosts  ││  254 hosts  │
     └─────────────┘└─────────────┘└─────────────┘
```

---

## 6. Practice Problems

### Problem 1: Basic Subnet Calculation

Calculate the network information for the following IP addresses.

**a) 192.168.50.100/27**
```
Network address: ____________
Broadcast address: ____________
Host range: ____________
Usable host count: ____________
```

**b) 10.20.30.200/21**
```
Network address: ____________
Broadcast address: ____________
Host range: ____________
Usable host count: ____________
```

**c) 172.31.128.50/18**
```
Network address: ____________
Broadcast address: ____________
Host range: ____________
Usable host count: ____________
```

### Problem 2: Subnet Division

**a)** Divide 192.168.100.0/24 into 8 equal subnets.

| Subnet | Network Address | Host Range | Broadcast |
|--------|----------------|-----------|-----------|
| 1      |                |           |           |
| 2      |                |           |           |
| 3      |                |           |           |
| 4      |                |           |           |
| 5      |                |           |           |
| 6      |                |           |           |
| 7      |                |           |           |
| 8      |                |           |           |

**b)** How many subnets supporting 16,000 hosts each can be created from 10.0.0.0/8?

### Problem 3: VLSM Design

**Scenario**: Divide 172.30.0.0/23 according to requirements.

| Network      | Required Hosts |
|-------------|----------------|
| LAN A       | 120            |
| LAN B       | 60             |
| LAN C       | 30             |
| LAN D       | 10             |
| WAN Link 1  | 2              |
| WAN Link 2  | 2              |

Design results:

| Network     | Subnet | Network Address | Host Range | Broadcast |
|------------|--------|----------------|-----------|-----------|
| LAN A      |        |                |           |           |
| LAN B      |        |                |           |           |
| LAN C      |        |                |           |           |
| LAN D      |        |                |           |           |
| WAN Link 1 |        |                |           |           |
| WAN Link 2 |        |                |           |           |

### Problem 4: Comprehensive Network Design

**Scenario**: You must design a network for a new company.

- Allocated network: 10.50.0.0/20
- Requirements:
  - Sales: 500 people
  - Technical: 250 people
  - Admin: 100 people
  - Server farm: 60 servers
  - DMZ: 20 servers
  - Expect 30% growth over next 3 years

Allocate appropriate subnets to each department and design for future expansion.

---

## Answers

### Problem 1 Answers

**a) 192.168.50.100/27**
```
Block size: 256 - 224 = 32
100 ÷ 32 = 3 (remainder 4) → Start: 96
Network address: 192.168.50.96
Broadcast address: 192.168.50.127
Host range: 192.168.50.97 - 192.168.50.126
Usable host count: 30
```

**b) 10.20.30.200/21**
```
/21 = 5 network bits in third octet, 3 bits + 8 bits host
Block size (3rd octet): 8
30 ÷ 8 = 3 (remainder 6) → Start: 24
Network address: 10.20.24.0
Broadcast address: 10.20.31.255
Host range: 10.20.24.1 - 10.20.31.254
Usable host count: 2,046
```

**c) 172.31.128.50/18**
```
/18 = 2 network bits in second octet
Block size (2nd octet): 64
128 ÷ 64 = 2 → Start: 128
Network address: 172.31.128.0
Broadcast address: 172.31.191.255
Host range: 172.31.128.1 - 172.31.191.254
Usable host count: 16,382
```

### Problem 2 Answers

**a) 8 subnet division**
```
8 = 2^3 → Need to add 3 bits
/24 + 3 = /27 (Block size: 32)
```

| Subnet | Network Address      | Host Range    | Broadcast        |
|--------|---------------------|---------------|------------------|
| 1      | 192.168.100.0/27    | .1 - .30      | 192.168.100.31   |
| 2      | 192.168.100.32/27   | .33 - .62     | 192.168.100.63   |
| 3      | 192.168.100.64/27   | .65 - .94     | 192.168.100.95   |
| 4      | 192.168.100.96/27   | .97 - .126    | 192.168.100.127  |
| 5      | 192.168.100.128/27  | .129 - .158   | 192.168.100.159  |
| 6      | 192.168.100.160/27  | .161 - .190   | 192.168.100.191  |
| 7      | 192.168.100.192/27  | .193 - .222   | 192.168.100.223  |
| 8      | 192.168.100.224/27  | .225 - .254   | 192.168.100.255  |

**b) 16,000 host subnets**
```
16,000 hosts → 2^14 = 16,384 needed → 14 host bits
CIDR: 32 - 14 = /18
10.0.0.0/8 → /18 subnet count: 2^(18-8) = 2^10 = 1,024
```

### Problem 3 Answer (VLSM)

| Network    | Subnet | Network Address  | Host Range           | Broadcast      |
|-----------|--------|-----------------|---------------------|----------------|
| LAN A     | /25    | 172.30.0.0/25   | 172.30.0.1 - .126   | 172.30.0.127   |
| LAN B     | /26    | 172.30.0.128/26 | 172.30.0.129 - .190 | 172.30.0.191   |
| LAN C     | /27    | 172.30.0.192/27 | 172.30.0.193 - .222 | 172.30.0.223   |
| LAN D     | /28    | 172.30.0.224/28 | 172.30.0.225 - .238 | 172.30.0.239   |
| WAN Link 1| /30    | 172.30.0.240/30 | 172.30.0.241 - .242 | 172.30.0.243   |
| WAN Link 2| /30    | 172.30.0.244/30 | 172.30.0.245 - .246 | 172.30.0.247   |

---

**Previous**: [IP Addressing and Subnetting](./06_IP_Address_Subnetting.md) | **Next**: [Routing Basics](./08_Routing_Basics.md)

---

## 7. References

### Online Tools

- [Subnet Calculator](https://www.subnet-calculator.com/)
- [Visual Subnet Calculator](https://www.davidc.net/sites/default/subnets/subnets.html)
- [IP Address Guide](https://www.ipaddressguide.com/cidr)

### Learning Materials

- RFC 950 - Internet Standard Subnetting Procedure
- RFC 1878 - Variable Length Subnet Table
- Cisco Networking Academy - Subnetting

### Command Reference

```bash
# Linux/macOS - Check network configuration
# ip addr: preferred over ifconfig — shows CIDR notation directly and supports newer features (VLANs, namespaces)
ip addr show
# ifconfig: legacy tool — still useful on macOS where 'ip' is not installed by default
ifconfig

# Subnet calculation tools — verify your manual calculations before deploying
ipcalc 192.168.1.0/24      # Outputs network/broadcast/host range — catches off-by-one errors
sipcalc 192.168.1.0/24     # More detailed: shows binary representation and wildcard mask

# Windows — /all flag shows subnet mask, gateway, DNS, and DHCP lease info (not just IP)
ipconfig /all
```

---

**Document Information**
- Last updated: 2024
- Difficulty: ⭐⭐⭐
- Estimated learning time: 3-4 hours
