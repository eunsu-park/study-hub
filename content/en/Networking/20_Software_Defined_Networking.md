[Previous: Container Networking](./19_Container_Networking.md) | [Next: Quality of Service](./21_Quality_of_Service.md)

---

# 20. Software-Defined Networking (SDN)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the control plane / data plane separation and why SDN was created
2. Describe the OpenFlow protocol and flow table matching pipeline
3. Compare SDN controller architectures (centralized vs distributed)
4. Implement a basic SDN controller that programs forwarding rules
5. Discuss P4 programmable data planes and their role in modern networking

---

## Table of Contents

1. [Traditional vs SDN Architecture](#1-traditional-vs-sdn-architecture)
2. [OpenFlow Protocol](#2-openflow-protocol)
3. [SDN Controllers](#3-sdn-controllers)
4. [Network Applications](#4-network-applications)
5. [P4: Programmable Data Planes](#5-p4-programmable-data-planes)
6. [SDN in Practice](#6-sdn-in-practice)
7. [Exercises](#7-exercises)

---

## 1. Traditional vs SDN Architecture

### 1.1 The Problem with Traditional Networking

In traditional networks, each device (router, switch) contains both:
- **Control plane**: Routing decisions (OSPF, BGP, spanning tree)
- **Data plane**: Packet forwarding based on control plane decisions

```
Traditional Network:

  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Router A │    │ Router B │    │ Router C │
  │┌────────┐│    │┌────────┐│    │┌────────┐│
  ││Control ││    ││Control ││    ││Control ││
  ││ Plane  ││    ││ Plane  ││    ││ Plane  ││
  │├────────┤│    │├────────┤│    │├────────┤│
  ││  Data  ││    ││  Data  ││    ││  Data  ││
  ││ Plane  ││    ││ Plane  ││    ││ Plane  ││
  │└────────┘│    │└────────┘│    │└────────┘│
  └──────────┘    └──────────┘    └──────────┘
  Each device independently decides how to forward.
  Configuration: CLI per device (error-prone, slow).
```

Problems:
- **Distributed complexity**: Each device runs its own protocols
- **Vendor lock-in**: Proprietary CLI and firmware per vendor
- **Slow innovation**: New features require firmware updates on all devices
- **Inconsistency**: Hard to enforce network-wide policies

### 1.2 SDN Architecture

SDN separates the control plane into a centralized controller:

```
SDN Architecture:

              ┌────────────────────────┐
              │     SDN Controller     │  ← Centralized brain
              │  (network-wide view)   │
              │                        │
              │  • Routing decisions   │
              │  • Policy enforcement  │
              │  • Topology discovery  │
              └───────┬────┬────┬──────┘
                      │    │    │   Southbound API (OpenFlow)
              ┌───────┘    │    └───────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │Switch A │ │Switch B │ │Switch C │
         │┌───────┐│ │┌───────┐│ │┌───────┐│
         ││ Data  ││ ││ Data  ││ ││ Data  ││
         ││ Plane ││ ││ Plane ││ ││ Plane ││
         │└───────┘│ │└───────┘│ │└───────┘│
         └─────────┘ └─────────┘ └─────────┘
         "Dumb" switches — just forward as told.
```

### 1.3 SDN Layers

| Layer | Function | Examples |
|-------|----------|---------|
| Application layer | Network apps (firewall, load balancer, monitor) | Custom apps, northbound API consumers |
| Control layer | Centralized logic, topology, state | OpenDaylight, ONOS, Ryu, Floodlight |
| Infrastructure layer | Packet forwarding hardware | OpenFlow switches, P4 switches |

APIs:
- **Northbound API**: Controller ↔ Applications (REST, gRPC)
- **Southbound API**: Controller ↔ Switches (OpenFlow, P4Runtime, NETCONF)

---

## 2. OpenFlow Protocol

### 2.1 Flow Tables

Each OpenFlow switch contains one or more **flow tables**. Each table has flow entries:

```
Flow Entry:
┌──────────────┬───────────┬──────────────┬─────────┬──────────┐
│ Match Fields │ Priority  │ Instructions │ Counters│ Timeouts │
│              │           │              │         │          │
│ src_ip       │ Higher =  │ Forward to   │ Packets │ Idle: 60s│
│ dst_ip       │ checked   │ port 3       │ Bytes   │ Hard: 0  │
│ src_port     │ first     │ Set VLAN     │         │          │
│ dst_port     │           │ Drop         │         │          │
│ protocol     │           │ Send to      │         │          │
│ in_port      │           │ controller   │         │          │
│ VLAN ID      │           │              │         │          │
└──────────────┴───────────┴──────────────┴─────────┴──────────┘
```

### 2.2 Packet Processing Pipeline

```
Packet arrives at switch:

  ┌─────────┐    ┌──────────┐    ┌──────────┐
  │ Table 0 │───►│ Table 1  │───►│ Table 2  │───► ...
  └────┬────┘    └────┬─────┘    └────┬─────┘
       │              │               │
  Match found?   Match found?    Match found?
       │              │               │
   Yes: execute   Yes: execute    Yes: execute
   instructions   instructions    instructions
       │              │               │
   No: go to      No: go to       No: table-miss
   next table     next table      → send to controller
                                    or drop
```

### 2.3 OpenFlow Messages

```python
# Conceptual OpenFlow message types

class OpenFlowMessages:
    """Key OpenFlow message categories."""

    # Controller → Switch
    FLOW_MOD = "flow_mod"           # Add/modify/delete flow entries
    PACKET_OUT = "packet_out"       # Send a packet out a specific port
    BARRIER = "barrier"             # Ensure all prior messages processed

    # Switch → Controller
    PACKET_IN = "packet_in"         # Packet didn't match any flow → ask controller
    FLOW_REMOVED = "flow_removed"   # Flow entry expired or deleted
    PORT_STATUS = "port_status"     # Port state change (up/down)

    # Symmetric
    HELLO = "hello"                 # Connection setup
    ECHO = "echo"                   # Keepalive
    FEATURES_REQUEST = "features"   # Controller asks switch capabilities
```

### 2.4 Flow Matching Example

```python
def match_packet(flow_tables, packet):
    """Simulate OpenFlow packet matching.

    Packets are matched against flow entries in priority order.
    First match wins within each table.
    """
    for table in flow_tables:
        matched_entry = None
        best_priority = -1

        for entry in table:
            if matches(packet, entry['match']) and entry['priority'] > best_priority:
                matched_entry = entry
                best_priority = entry['priority']

        if matched_entry:
            # Execute instructions
            for instruction in matched_entry['instructions']:
                if instruction['type'] == 'output':
                    return {'action': 'forward', 'port': instruction['port']}
                elif instruction['type'] == 'goto_table':
                    break  # continue to specified table
                elif instruction['type'] == 'drop':
                    return {'action': 'drop'}

    # No match in any table → table-miss
    return {'action': 'send_to_controller'}
```

---

## 3. SDN Controllers

### 3.1 Controller Architecture

```
┌───────────────────────────────────────────────────┐
│                  SDN Controller                    │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │         Northbound API (REST/gRPC)          │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ Topology │  │ Device   │  │ Flow Rule     │   │
│  │ Manager  │  │ Manager  │  │ Manager       │   │
│  └──────────┘  └──────────┘  └───────────────┘   │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ Stats    │  │ Path     │  │ Host          │   │
│  │ Collector│  │ Compute  │  │ Tracker       │   │
│  └──────────┘  └──────────┘  └───────────────┘   │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │      Southbound API (OpenFlow/P4Runtime)    │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

### 3.2 Major SDN Controllers

| Controller | Language | Key Features | Use Case |
|-----------|----------|-------------|----------|
| OpenDaylight (ODL) | Java | Modular, YANG models, NETCONF/RESTCONF | Enterprise, service provider |
| ONOS | Java | Distributed, intent-based, high availability | Carrier-grade, WAN |
| Ryu | Python | Lightweight, easy to learn, component-based | Research, prototyping |
| Floodlight | Java | REST API, OpenStack integration | Cloud networking |
| FAUCET | Python | Production-ready, OpenFlow 1.3, YAML config | Campus, enterprise |

### 3.3 Simple Controller Logic

```python
class SimpleSDNController:
    """Minimal SDN controller demonstrating reactive forwarding.

    When a switch receives a packet with no matching flow,
    it sends the packet to the controller (PACKET_IN).
    The controller decides what to do and installs a flow rule.
    """

    def __init__(self):
        self.mac_table = {}  # switch_id → {mac → port}
        self.topology = {}   # switch_id → {port → neighbor}

    def handle_packet_in(self, switch_id, in_port, packet):
        """Handle a packet that didn't match any flow rule.

        This is the core of reactive forwarding:
        1. Learn the source MAC → port mapping
        2. Look up destination MAC
        3. Either forward to known port or flood
        """
        src_mac = packet['src_mac']
        dst_mac = packet['dst_mac']

        # Learn source MAC address
        if switch_id not in self.mac_table:
            self.mac_table[switch_id] = {}
        self.mac_table[switch_id][src_mac] = in_port

        # Lookup destination
        if dst_mac in self.mac_table.get(switch_id, {}):
            out_port = self.mac_table[switch_id][dst_mac]
            # Install flow rule so future packets go directly
            self.install_flow(switch_id, dst_mac, out_port)
            return {'action': 'forward', 'port': out_port}
        else:
            # Unknown destination → flood to all ports except source
            return {'action': 'flood', 'exclude_port': in_port}

    def install_flow(self, switch_id, dst_mac, out_port):
        """Install a forwarding rule on the switch."""
        flow_rule = {
            'match': {'dst_mac': dst_mac},
            'priority': 100,
            'instructions': [{'type': 'output', 'port': out_port}],
            'idle_timeout': 300,  # remove after 5 min idle
        }
        print(f"  Installing flow on switch {switch_id}: "
              f"{dst_mac} → port {out_port}")
        return flow_rule
```

---

## 4. Network Applications

### 4.1 SDN-Based Firewall

```python
class SDNFirewall:
    """Stateless firewall implemented as an SDN application.

    Installs flow rules that block or allow traffic based on
    IP addresses, ports, and protocols.
    """

    def __init__(self, controller):
        self.controller = controller
        self.rules = []

    def add_rule(self, src_ip=None, dst_ip=None, protocol=None,
                 dst_port=None, action='deny'):
        """Add a firewall rule."""
        self.rules.append({
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'dst_port': dst_port,
            'action': action,
        })

    def evaluate(self, packet):
        """Check packet against firewall rules (first match wins)."""
        for rule in self.rules:
            if self._matches(packet, rule):
                return rule['action']
        return 'allow'  # default allow

    @staticmethod
    def _matches(packet, rule):
        for field in ['src_ip', 'dst_ip', 'protocol', 'dst_port']:
            if rule[field] is not None and packet.get(field) != rule[field]:
                return False
        return True
```

### 4.2 SDN Load Balancer

```python
class SDNLoadBalancer:
    """Round-robin load balancer as an SDN application.

    Distributes incoming connections across backend servers
    by rewriting destination IP/port in flow rules.
    """

    def __init__(self, vip, backends):
        """
        vip: Virtual IP address (what clients connect to)
        backends: List of {'ip': ..., 'port': ..., 'weight': ...}
        """
        self.vip = vip
        self.backends = backends
        self.current_idx = 0

    def select_backend(self):
        """Round-robin backend selection."""
        backend = self.backends[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.backends)
        return backend

    def create_flow_rules(self, client_ip, client_port):
        """Create bidirectional flow rules for a new connection."""
        backend = self.select_backend()

        # Forward direction: client → VIP becomes client → backend
        forward_rule = {
            'match': {
                'src_ip': client_ip,
                'dst_ip': self.vip,
                'src_port': client_port,
            },
            'instructions': [
                {'type': 'set_field', 'field': 'dst_ip',
                 'value': backend['ip']},
                {'type': 'set_field', 'field': 'dst_port',
                 'value': backend['port']},
                {'type': 'output', 'port': 'computed'},
            ],
        }

        # Reverse direction: backend → client becomes VIP → client
        reverse_rule = {
            'match': {
                'src_ip': backend['ip'],
                'dst_ip': client_ip,
                'dst_port': client_port,
            },
            'instructions': [
                {'type': 'set_field', 'field': 'src_ip',
                 'value': self.vip},
                {'type': 'output', 'port': 'computed'},
            ],
        }

        return forward_rule, reverse_rule
```

---

## 5. P4: Programmable Data Planes

### 5.1 Beyond OpenFlow

OpenFlow has a fixed set of match fields and actions. **P4** (Programming Protocol-independent Packet Processors) lets you define custom packet formats and processing logic.

```
OpenFlow:                           P4:
  Fixed header fields               Custom header definitions
  Fixed match/action pipeline        Programmable parser
  New protocols → new OF version     New protocols → new P4 program
```

### 5.2 P4 Program Structure

```
P4 program:

  ┌─────────────────────────────────────┐
  │  1. Header Definitions              │
  │     Define packet header formats    │
  │     (Ethernet, IPv4, custom, ...)   │
  │                                     │
  │  2. Parser                          │
  │     Extract headers from packets    │
  │     (state machine)                 │
  │                                     │
  │  3. Match-Action Tables             │
  │     Define tables + actions         │
  │     (like OpenFlow but custom)      │
  │                                     │
  │  4. Control Flow                    │
  │     Apply tables in order           │
  │     (if/else, table chains)         │
  │                                     │
  │  5. Deparser                        │
  │     Reassemble packet with          │
  │     modified headers                │
  └─────────────────────────────────────┘
```

### 5.3 P4 Example (Conceptual)

```
// Define custom header for in-network telemetry
header telemetry_t {
    bit<32> switch_id;
    bit<32> ingress_port;
    bit<48> ingress_timestamp;
    bit<32> queue_depth;
}

// Each switch adds its telemetry data to the packet
// By the time the packet reaches the destination,
// it contains per-hop telemetry from every switch on the path
```

---

## 6. SDN in Practice

### 6.1 Deployment Models

| Model | Description | Example |
|-------|-------------|---------|
| Campus SDN | Centralized switch management | Cisco DNA Center, Aruba Central |
| Data center SDN | Virtual networking overlays | VMware NSX, Cisco ACI |
| WAN SDN (SD-WAN) | Software-defined wide area network | Cisco Viptela, VMware VeloCloud |
| Carrier SDN | Service provider network control | ONOS, OpenDaylight |

### 6.2 SDN vs Traditional: Trade-offs

| Aspect | Traditional | SDN |
|--------|------------|-----|
| Control | Distributed, autonomous | Centralized, programmatic |
| Scalability | Each device scales independently | Controller is a potential bottleneck |
| Failure mode | Graceful degradation | Controller failure = network blind |
| Innovation speed | Slow (vendor firmware) | Fast (software-defined) |
| Operational cost | High (per-device CLI) | Lower (automation, APIs) |
| Vendor flexibility | Locked to vendor ecosystem | Open standards possible |

### 6.3 High Availability

```
Controller HA approaches:

  1. Active-Standby:
     ┌────────────┐     ┌────────────┐
     │ Controller │────►│ Controller │
     │  (Active)  │     │ (Standby)  │
     └────────────┘     └────────────┘
     Simple but wastes standby resources.

  2. Clustered (ONOS model):
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ Controller │  │ Controller │  │ Controller │
     │   Node 1   │──│   Node 2   │──│   Node 3   │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
     ┌─────┴──┐      ┌─────┴──┐      ┌─────┴──┐
     │Switches│      │Switches│      │Switches│
     │ zone 1 │      │ zone 2 │      │ zone 3 │
     └────────┘      └────────┘      └────────┘
     Each controller manages its zone, shares state via Raft.
```

---

## 7. Exercises

### Exercise 1: Reactive L2 Learning Switch

Implement a reactive L2 learning switch controller:
1. Maintain a MAC address table per switch
2. On PACKET_IN: learn source MAC, lookup destination
3. If destination known → install flow rule, forward
4. If unknown → flood to all ports except ingress
5. Add idle timeout (60s) to flow rules
6. Test with a 4-switch linear topology

### Exercise 2: SDN Firewall

Build an SDN firewall application:
1. Define ACL rules (allow/deny by src_ip, dst_ip, protocol, port)
2. Install proactive flow rules for allowed traffic
3. Default deny: packets not matching any allow rule are dropped
4. Add logging: count blocked packets per rule
5. Implement rule priority ordering

### Exercise 3: Shortest Path Routing

Implement shortest-path forwarding:
1. Build a topology graph from switch connections (LLDP discovery)
2. Compute shortest paths between all host pairs (Dijkstra or BFS)
3. Install proactive flow rules along computed paths
4. Handle topology changes: recalculate paths when a link goes down
5. Compare reactive vs proactive installation latency

### Exercise 4: SDN Load Balancer

Create a round-robin load balancer:
1. Virtual IP (VIP) → 3 backend servers
2. New connections: select backend, install rewrite rules
3. Existing connections: use installed rules (no controller involvement)
4. Add health checking: remove failed backends from rotation
5. Measure the flow rule installation overhead

### Exercise 5: Controller Scalability

Analyze controller scalability:
1. Simulate increasing numbers of switches (10, 50, 100, 500)
2. Generate PACKET_IN events at various rates
3. Measure controller response time vs load
4. Identify the bottleneck: CPU, memory, or southbound bandwidth
5. Discuss strategies to scale beyond a single controller

---

*End of Lesson 20*
