"""
Zigbee Mesh Network Simulator

Simulates a Zigbee mesh network with:
- Device roles: Coordinator, Router, End Device
- AODV-style mesh routing with route discovery
- Self-healing when routers fail
- Message delivery with hop tracking

This simulator demonstrates mesh topology concepts without
requiring actual Zigbee hardware.
"""

import random
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeviceRole(Enum):
    COORDINATOR = "Coordinator"
    ROUTER = "Router"
    END_DEVICE = "EndDevice"


@dataclass
class ZigbeeDevice:
    """A device in the Zigbee mesh network."""
    addr: int                           # 16-bit network address
    role: DeviceRole
    x: float                            # Position for range calculation
    y: float
    parent: Optional[int] = None        # Parent router address (for end devices)
    tx_power_dbm: float = 0.0           # Transmit power in dBm
    rx_sensitivity_dbm: float = -95.0   # Minimum receivable signal
    active: bool = True


@dataclass
class Message:
    """A message traveling through the mesh."""
    src: int
    dst: int
    payload: str
    hops: list = field(default_factory=list)
    delivered: bool = False


class ZigbeeMeshNetwork:
    """Simulates a Zigbee mesh network with AODV-style routing."""

    COORDINATOR_ADDR = 0x0000

    def __init__(self, max_range_m: float = 30.0, pan_id: int = 0x1234):
        self.pan_id = pan_id
        self.max_range_m = max_range_m
        self.devices: dict[int, ZigbeeDevice] = {}
        self.routing_tables: dict[int, dict[int, int]] = defaultdict(dict)
        self._next_addr = 1
        self.message_log: list[Message] = []

    def _distance(self, d1: ZigbeeDevice, d2: ZigbeeDevice) -> float:
        return math.sqrt((d1.x - d2.x) ** 2 + (d1.y - d2.y) ** 2)

    def _in_range(self, d1: ZigbeeDevice, d2: ZigbeeDevice) -> bool:
        return self._distance(d1, d2) <= self.max_range_m

    def _get_neighbors(self, addr: int) -> list[int]:
        """Get all active devices within radio range."""
        device = self.devices[addr]
        neighbors = []
        for other_addr, other in self.devices.items():
            if other_addr != addr and other.active and self._in_range(device, other):
                # End devices can only talk to their parent router
                if device.role == DeviceRole.END_DEVICE and other_addr != device.parent:
                    continue
                if other.role == DeviceRole.END_DEVICE and addr != other.parent:
                    continue
                neighbors.append(other_addr)
        return neighbors

    def add_coordinator(self, x: float, y: float) -> int:
        """Add the network coordinator (one per network)."""
        addr = self.COORDINATOR_ADDR
        self.devices[addr] = ZigbeeDevice(
            addr=addr, role=DeviceRole.COORDINATOR, x=x, y=y
        )
        print(f"  [Coordinator] Address 0x{addr:04X} at ({x:.0f}, {y:.0f})")
        return addr

    def add_router(self, x: float, y: float) -> int:
        """Add a router to the network."""
        addr = self._next_addr
        self._next_addr += 1

        # Find a parent (coordinator or existing router) in range
        device = ZigbeeDevice(addr=addr, role=DeviceRole.ROUTER, x=x, y=y)
        parent = None
        for existing_addr, existing in self.devices.items():
            if existing.active and existing.role in (DeviceRole.COORDINATOR, DeviceRole.ROUTER):
                if self._in_range(device, existing):
                    parent = existing_addr
                    break

        if parent is None:
            print(f"  [Router] Address 0x{addr:04X} — no parent in range, joining failed")
            return -1

        device.parent = parent
        self.devices[addr] = device
        print(f"  [Router] Address 0x{addr:04X} at ({x:.0f}, {y:.0f}), "
              f"parent=0x{parent:04X}")
        return addr

    def add_end_device(self, x: float, y: float) -> int:
        """Add an end device (battery-powered, sleeps)."""
        addr = self._next_addr
        self._next_addr += 1

        device = ZigbeeDevice(addr=addr, role=DeviceRole.END_DEVICE, x=x, y=y)

        # Find nearest router/coordinator in range
        best_parent = None
        best_dist = float('inf')
        for existing_addr, existing in self.devices.items():
            if existing.active and existing.role in (DeviceRole.COORDINATOR, DeviceRole.ROUTER):
                if self._in_range(device, existing):
                    d = self._distance(device, existing)
                    if d < best_dist:
                        best_dist = d
                        best_parent = existing_addr

        if best_parent is None:
            print(f"  [EndDevice] Address 0x{addr:04X} — no router in range")
            return -1

        device.parent = best_parent
        self.devices[addr] = device
        print(f"  [EndDevice] Address 0x{addr:04X} at ({x:.0f}, {y:.0f}), "
              f"parent=0x{best_parent:04X}")
        return addr

    def discover_route(self, src: int, dst: int) -> Optional[list[int]]:
        """AODV-style route discovery using BFS."""
        if src == dst:
            return [src]

        visited = {src}
        queue = deque([(src, [src])])

        while queue:
            current, path = queue.popleft()
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    if neighbor == dst:
                        # Cache route
                        for i in range(len(new_path) - 1):
                            self.routing_tables[new_path[i]][dst] = new_path[i + 1]
                        return new_path
                    queue.append((neighbor, new_path))

        return None  # No route found

    def send_message(self, src: int, dst: int, payload: str) -> Message:
        """Send a message through the mesh network."""
        msg = Message(src=src, dst=dst, payload=payload)

        route = self.discover_route(src, dst)
        if route is None:
            print(f"  [SEND] 0x{src:04X} → 0x{dst:04X}: NO ROUTE FOUND")
            self.message_log.append(msg)
            return msg

        msg.hops = route
        msg.delivered = True
        hop_str = " → ".join(f"0x{h:04X}" for h in route)
        print(f"  [SEND] {payload}")
        print(f"         Route: {hop_str} ({len(route)-1} hops)")
        self.message_log.append(msg)
        return msg

    def fail_device(self, addr: int):
        """Simulate a device failure (power loss, hardware fault)."""
        if addr in self.devices:
            self.devices[addr].active = False
            # Clear routing entries through this device
            for table in self.routing_tables.values():
                to_remove = [dst for dst, nxt in table.items() if nxt == addr]
                for dst in to_remove:
                    del table[dst]
            print(f"  [FAIL] Device 0x{addr:04X} ({self.devices[addr].role.value}) "
                  f"has failed!")

    def restore_device(self, addr: int):
        """Restore a failed device."""
        if addr in self.devices:
            self.devices[addr].active = True
            print(f"  [RESTORE] Device 0x{addr:04X} is back online")

    def print_topology(self):
        """Print network topology as adjacency list."""
        print("\n  Network Topology:")
        for addr, device in sorted(self.devices.items()):
            if not device.active:
                continue
            neighbors = self._get_neighbors(addr)
            role = device.role.value[0]  # C/R/E
            n_str = ", ".join(f"0x{n:04X}" for n in neighbors)
            status = "" if device.active else " [OFFLINE]"
            print(f"    0x{addr:04X} [{role}] ({device.x:.0f},{device.y:.0f}) "
                  f"→ [{n_str}]{status}")


def demo_mesh_network():
    """Demonstrate Zigbee mesh networking with routing and self-healing."""
    print("=" * 60)
    print("Zigbee Mesh Network Simulation")
    print("=" * 60)

    net = ZigbeeMeshNetwork(max_range_m=25)

    # Create a house-like layout (2 floors)
    print("\n--- Network Formation ---")
    coord = net.add_coordinator(x=50, y=50)   # Living room hub

    # Ground floor routers (mains-powered devices like smart plugs)
    r1 = net.add_router(x=20, y=50)  # Kitchen
    r2 = net.add_router(x=80, y=50)  # Bedroom
    r3 = net.add_router(x=50, y=20)  # Hallway

    # Second floor router (relayed through ground floor)
    r4 = net.add_router(x=50, y=80)  # Upstairs

    # End devices (battery sensors)
    e1 = net.add_end_device(x=10, y=50)  # Kitchen temp sensor
    e2 = net.add_end_device(x=90, y=50)  # Bedroom motion sensor
    e3 = net.add_end_device(x=50, y=90)  # Upstairs humidity sensor

    net.print_topology()

    # Send messages
    print("\n--- Message Delivery ---")
    net.send_message(e1, coord, "Kitchen temp: 23.5°C")
    net.send_message(e2, e3, "Motion detected in bedroom")
    net.send_message(coord, e3, "Set thermostat to 22°C")

    # Simulate router failure
    print("\n--- Self-Healing Test ---")
    net.fail_device(r1)
    print("\n  Attempting to send from kitchen sensor after R1 failure:")
    msg = net.send_message(e1, coord, "Kitchen temp: 24.0°C")
    if not msg.delivered:
        print("  → Kitchen sensor lost connectivity (parent router failed)")
        print("  → In real Zigbee, the sensor would scan for a new parent")

    # Test alternative routing
    print("\n  Testing route between remaining devices:")
    net.send_message(e2, e3, "Bedroom light turned off")

    # Restore and test
    print("\n--- Device Restoration ---")
    net.restore_device(r1)
    net.send_message(e1, coord, "Kitchen temp: 24.1°C (reconnected)")

    # Statistics
    print("\n--- Network Statistics ---")
    total = len(net.message_log)
    delivered = sum(1 for m in net.message_log if m.delivered)
    avg_hops = sum(len(m.hops) - 1 for m in net.message_log if m.delivered)
    if delivered > 0:
        avg_hops /= delivered
    print(f"  Messages sent:     {total}")
    print(f"  Messages delivered: {delivered}")
    print(f"  Delivery rate:     {delivered/total*100:.0f}%")
    print(f"  Average hops:      {avg_hops:.1f}")


def demo_protocol_comparison():
    """Compare wireless protocol specifications."""
    print("\n" + "=" * 60)
    print("Protocol Comparison Matrix")
    print("=" * 60)

    protocols = [
        {"name": "Zigbee", "freq": "2.4 GHz", "rate": "250 kbps",
         "range": "10-30m", "devices": "65,000", "battery": "Years",
         "mesh": "Yes", "ip": "No"},
        {"name": "Z-Wave", "freq": "908 MHz", "rate": "100 kbps",
         "range": "30-100m", "devices": "232", "battery": "Years",
         "mesh": "Yes", "ip": "No"},
        {"name": "Thread", "freq": "2.4 GHz", "rate": "250 kbps",
         "range": "10-30m", "devices": "250+", "battery": "Years",
         "mesh": "Yes", "ip": "Yes (IPv6)"},
        {"name": "BLE", "freq": "2.4 GHz", "rate": "2 Mbps",
         "range": "10-30m", "devices": "7/piconet", "battery": "Months-Years",
         "mesh": "BLE Mesh", "ip": "No"},
        {"name": "WiFi", "freq": "2.4/5 GHz", "rate": "54-600 Mbps",
         "range": "30-50m", "devices": "~32", "battery": "Hours-Days",
         "mesh": "WiFi Mesh", "ip": "Yes (IPv4/6)"},
    ]

    header = f"  {'Protocol':<10} {'Frequency':<12} {'Rate':<12} {'Range':<10} " \
             f"{'Devices':<10} {'Battery':<12} {'Mesh':<10} {'IP':<12}"
    print(f"\n{header}")
    print("  " + "-" * 88)
    for p in protocols:
        print(f"  {p['name']:<10} {p['freq']:<12} {p['rate']:<12} "
              f"{p['range']:<10} {p['devices']:<10} {p['battery']:<12} "
              f"{p['mesh']:<10} {p['ip']:<12}")


if __name__ == "__main__":
    demo_mesh_network()
    demo_protocol_comparison()
