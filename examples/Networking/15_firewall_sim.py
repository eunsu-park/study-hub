"""
Firewall Simulator

Demonstrates:
- Packet-filter firewall rules
- Stateful vs stateless filtering
- Rule ordering and first-match semantics
- Default deny/allow policies

Theory:
- Firewalls control network traffic based on rules.
- Packet filter: inspects individual packets (src/dst IP, port,
  protocol). Stateless — doesn't track connections.
- Stateful firewall: tracks connection state. Allows return
  traffic for established connections automatically.
- Rules are evaluated top-down; first match wins.
- Default policy: ACCEPT or DROP for unmatched packets.

Adapted from Networking Lesson 15.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Action(Enum):
    ACCEPT = "ACCEPT"
    DROP = "DROP"
    LOG = "LOG"


class Protocol(Enum):
    TCP = "TCP"
    UDP = "UDP"
    ICMP = "ICMP"
    ANY = "ANY"


@dataclass
class Packet:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: Protocol
    direction: str = "IN"  # IN or OUT

    def __str__(self) -> str:
        return (f"{self.direction} {self.protocol.value} "
                f"{self.src_ip}:{self.src_port} → "
                f"{self.dst_ip}:{self.dst_port}")


@dataclass
class FirewallRule:
    name: str
    action: Action
    protocol: Protocol = Protocol.ANY
    src_ip: str = "any"
    dst_ip: str = "any"
    src_port: int | str = "any"
    dst_port: int | str = "any"
    direction: str = "any"  # IN, OUT, any

    # Why: Each field uses "any" as a wildcard to match all values. This
    # multi-field AND logic means a rule matches only when ALL specified
    # criteria match — just like iptables chains where each -s, -d, --dport
    # flag narrows the match further.
    def matches(self, packet: Packet) -> bool:
        if self.direction != "any" and self.direction != packet.direction:
            return False
        if self.protocol != Protocol.ANY and self.protocol != packet.protocol:
            return False
        if self.src_ip != "any" and not self._match_ip(packet.src_ip, self.src_ip):
            return False
        if self.dst_ip != "any" and not self._match_ip(packet.dst_ip, self.dst_ip):
            return False
        if self.src_port != "any" and not self._match_port(packet.src_port, self.src_port):
            return False
        if self.dst_port != "any" and not self._match_port(packet.dst_port, self.dst_port):
            return False
        return True

    def _match_ip(self, packet_ip: str, rule_ip: str) -> bool:
        if "/" in rule_ip:
            # CIDR match
            import struct, socket
            net_str, prefix_str = rule_ip.split("/")
            prefix = int(prefix_str)
            mask = (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF
            net = struct.unpack("!I", socket.inet_aton(net_str))[0] & mask
            ip = struct.unpack("!I", socket.inet_aton(packet_ip))[0] & mask
            return net == ip
        return packet_ip == rule_ip

    def _match_port(self, packet_port: int, rule_port: int | str) -> bool:
        if isinstance(rule_port, str) and "-" in rule_port:
            low, high = rule_port.split("-")
            return int(low) <= packet_port <= int(high)
        return packet_port == int(rule_port)

    def __str__(self) -> str:
        return (f"{self.action.value:<8} {self.protocol.value:<5} "
                f"{self.src_ip}:{self.src_port} → "
                f"{self.dst_ip}:{self.dst_port} [{self.direction}]"
                f" ({self.name})")


class Firewall:
    """Packet-filter firewall."""

    def __init__(self, default_action: Action = Action.DROP):
        self.rules: list[FirewallRule] = []
        self.default_action = default_action
        self.log: list[tuple[Packet, Action, str]] = []

    def add_rule(self, rule: FirewallRule) -> None:
        self.rules.append(rule)

    # Why: First-match semantics (not best-match) is the standard for firewalls.
    # This means rule ordering is critical — a broad ACCEPT placed before a
    # specific DROP will override the block. This matches iptables, pf, and
    # most commercial firewall behavior.
    def evaluate(self, packet: Packet) -> Action:
        """Evaluate packet against rules (first-match)."""
        for rule in self.rules:
            if rule.matches(packet):
                self.log.append((packet, rule.action, rule.name))
                return rule.action

        # Default policy
        self.log.append((packet, self.default_action, "DEFAULT"))
        return self.default_action

    def print_rules(self) -> None:
        print(f"    {'#':>3}  {'Action':<8} {'Proto':<5} "
              f"{'Source':<22} {'Destination':<22} {'Dir':<4} Name")
        print(f"    {'-'*3}  {'-'*8} {'-'*5} {'-'*22} {'-'*22} {'-'*4} {'-'*15}")
        for i, rule in enumerate(self.rules, 1):
            src = f"{rule.src_ip}:{rule.src_port}"
            dst = f"{rule.dst_ip}:{rule.dst_port}"
            print(f"    {i:>3}  {rule.action.value:<8} {rule.protocol.value:<5} "
                  f"{src:<22} {dst:<22} {rule.direction:<4} {rule.name}")
        print(f"    {'*':>3}  {self.default_action.value:<8} "
              f"{'ANY':<5} {'any:any':<22} {'any:any':<22} {'any':<4} DEFAULT")


# Why: Stateful inspection was a breakthrough over packet filtering. A
# stateless firewall cannot distinguish a legitimate HTTP response from
# an unsolicited inbound packet — both have src_port 80. By tracking
# connection state, the stateful firewall automatically allows return
# traffic for outbound connections without explicit inbound rules.
class StatefulFirewall(Firewall):
    """Stateful firewall that tracks connections."""

    def __init__(self, default_action: Action = Action.DROP):
        super().__init__(default_action)
        # Why: The connection table stores 5-tuples (src_ip, src_port, dst_ip,
        # dst_port, protocol) to uniquely identify each connection. Checking the
        # reversed tuple for inbound packets identifies return traffic.
        self.connections: set[tuple[str, int, str, int, str]] = set()

    def evaluate(self, packet: Packet) -> Action:
        # Check if this is a return packet for an established connection
        conn_key = (packet.dst_ip, packet.dst_port,
                    packet.src_ip, packet.src_port,
                    packet.protocol.value)
        if conn_key in self.connections:
            self.log.append((packet, Action.ACCEPT, "ESTABLISHED"))
            return Action.ACCEPT

        # Evaluate rules
        action = super().evaluate(packet)

        # Track new connections
        if action == Action.ACCEPT:
            fwd_key = (packet.src_ip, packet.src_port,
                       packet.dst_ip, packet.dst_port,
                       packet.protocol.value)
            self.connections.add(fwd_key)

        return action


# ── Demos ──────────────────────────────────────────────────────────────

def demo_basic_firewall():
    print("=" * 60)
    print("PACKET-FILTER FIREWALL")
    print("=" * 60)

    # Why: Default-deny (DROP) is the security best practice — only explicitly
    # allowed traffic passes through. This is far safer than default-accept,
    # where a missing rule means an open door.
    fw = Firewall(default_action=Action.DROP)

    # Common rules
    fw.add_rule(FirewallRule("Allow HTTP", Action.ACCEPT, Protocol.TCP,
                             dst_port=80, direction="IN"))
    fw.add_rule(FirewallRule("Allow HTTPS", Action.ACCEPT, Protocol.TCP,
                             dst_port=443, direction="IN"))
    fw.add_rule(FirewallRule("Allow SSH", Action.ACCEPT, Protocol.TCP,
                             src_ip="10.0.0.0/8", dst_port=22, direction="IN"))
    fw.add_rule(FirewallRule("Allow DNS", Action.ACCEPT, Protocol.UDP,
                             dst_port=53, direction="OUT"))
    fw.add_rule(FirewallRule("Block Telnet", Action.DROP, Protocol.TCP,
                             dst_port=23))
    fw.add_rule(FirewallRule("Allow outbound", Action.ACCEPT, direction="OUT"))

    print(f"\n  Firewall rules (default: DROP):")
    fw.print_rules()

    # Test packets
    packets = [
        Packet("203.0.113.1", "192.168.1.10", 54321, 80, Protocol.TCP, "IN"),
        Packet("203.0.113.1", "192.168.1.10", 54321, 443, Protocol.TCP, "IN"),
        Packet("10.0.0.5", "192.168.1.10", 54321, 22, Protocol.TCP, "IN"),
        Packet("203.0.113.1", "192.168.1.10", 54321, 22, Protocol.TCP, "IN"),
        Packet("203.0.113.1", "192.168.1.10", 54321, 23, Protocol.TCP, "IN"),
        Packet("192.168.1.10", "8.8.8.8", 12345, 53, Protocol.UDP, "OUT"),
        Packet("203.0.113.1", "192.168.1.10", 54321, 3306, Protocol.TCP, "IN"),
    ]

    print(f"\n  Packet evaluation:")
    print(f"    {'Packet':<55} {'Action':<8} Rule")
    print(f"    {'-'*55} {'-'*8} {'-'*15}")
    for pkt in packets:
        action = fw.evaluate(pkt)
        _, _, rule_name = fw.log[-1]
        print(f"    {str(pkt):<55} {action.value:<8} {rule_name}")


def demo_stateful():
    print("\n" + "=" * 60)
    print("STATEFUL vs STATELESS FIREWALL")
    print("=" * 60)

    # Stateless: must explicitly allow return traffic
    fw_stateless = Firewall(default_action=Action.DROP)
    fw_stateless.add_rule(FirewallRule("Allow HTTP out", Action.ACCEPT,
                                        Protocol.TCP, dst_port=80,
                                        direction="OUT"))
    # Without this rule, return traffic would be dropped!
    # fw_stateless.add_rule(FirewallRule("Allow HTTP return", ...))

    # Stateful: automatically allows return traffic
    fw_stateful = StatefulFirewall(default_action=Action.DROP)
    fw_stateful.add_rule(FirewallRule("Allow HTTP out", Action.ACCEPT,
                                       Protocol.TCP, dst_port=80,
                                       direction="OUT"))

    # Outbound request
    outbound = Packet("192.168.1.10", "93.184.216.34", 54321, 80,
                      Protocol.TCP, "OUT")
    # Return traffic
    inbound = Packet("93.184.216.34", "192.168.1.10", 80, 54321,
                     Protocol.TCP, "IN")

    print(f"\n  Outbound: {outbound}")
    print(f"  Return:   {inbound}")

    out_sl = fw_stateless.evaluate(outbound)
    in_sl = fw_stateless.evaluate(inbound)
    out_sf = fw_stateful.evaluate(outbound)
    in_sf = fw_stateful.evaluate(inbound)

    print(f"\n  {'Packet':<12} {'Stateless':<12} {'Stateful'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Outbound':<12} {out_sl.value:<12} {out_sf.value}")
    print(f"  {'Return':<12} {in_sl.value:<12} {in_sf.value}")

    print(f"\n  Stateless: blocks return traffic (no matching inbound rule)")
    print(f"  Stateful:  allows return traffic (tracked connection)")


def demo_rule_ordering():
    print("\n" + "=" * 60)
    print("RULE ORDERING MATTERS")
    print("=" * 60)

    # Order 1: specific before general
    fw1 = Firewall(default_action=Action.DROP)
    fw1.add_rule(FirewallRule("Block bad IP", Action.DROP, Protocol.TCP,
                              src_ip="203.0.113.100", dst_port=80))
    fw1.add_rule(FirewallRule("Allow HTTP", Action.ACCEPT, Protocol.TCP,
                              dst_port=80))

    # Order 2: general before specific (wrong!)
    fw2 = Firewall(default_action=Action.DROP)
    fw2.add_rule(FirewallRule("Allow HTTP", Action.ACCEPT, Protocol.TCP,
                              dst_port=80))
    fw2.add_rule(FirewallRule("Block bad IP", Action.DROP, Protocol.TCP,
                              src_ip="203.0.113.100", dst_port=80))

    pkt = Packet("203.0.113.100", "192.168.1.10", 54321, 80, Protocol.TCP, "IN")

    print(f"\n  Packet from blocked IP to port 80:")

    result1 = fw1.evaluate(pkt)
    _, _, rule1 = fw1.log[-1]
    result2 = fw2.evaluate(pkt)
    _, _, rule2 = fw2.log[-1]

    print(f"\n  Order 1 (specific first):")
    print(f"    Rule: {rule1} → {result1.value}")

    print(f"\n  Order 2 (general first):")
    print(f"    Rule: {rule2} → {result2.value} (BAD! Block rule never reached)")

    print(f"\n  Rule: Always place specific rules before general rules!")


if __name__ == "__main__":
    demo_basic_firewall()
    demo_stateful()
    demo_rule_ordering()
