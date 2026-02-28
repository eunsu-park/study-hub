"""
VPC (Virtual Private Cloud) Networking Simulation

Demonstrates cloud networking fundamentals:
- CIDR block allocation and subnet division
- Public vs private subnets with routing tables
- Security group rules (stateful firewall)
- Network ACLs (stateless firewall)
- Packet flow simulation through VPC components

No cloud account required -- all behavior is simulated locally.
"""

import ipaddress
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SubnetType(Enum):
    PUBLIC = "public"    # Has route to Internet Gateway
    PRIVATE = "private"  # No direct internet access; uses NAT Gateway for outbound


@dataclass
class CIDRBlock:
    """Represents a CIDR block (e.g., 10.0.0.0/16).
    CIDR (Classless Inter-Domain Routing) is how cloud providers allocate
    IP ranges. The /N suffix indicates how many bits are the network prefix,
    leaving (32-N) bits for host addresses."""
    cidr: str

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(self.cidr, strict=False)

    @property
    def total_ips(self) -> int:
        return self.network.num_addresses

    @property
    def usable_ips(self) -> int:
        """AWS reserves 5 IPs per subnet: network, router, DNS, future, broadcast.
        This is a common gotcha -- a /28 subnet has 16 IPs but only 11 usable."""
        return max(0, self.total_ips - 5)

    def subdivide(self, new_prefix: int) -> List[str]:
        """Split a CIDR block into smaller subnets.
        E.g., 10.0.0.0/16 -> four /18 subnets (one per AZ)."""
        return [str(subnet) for subnet in self.network.subnets(new_prefix=new_prefix)]


@dataclass
class RouteEntry:
    """A single route in a routing table.
    Routes determine where network traffic is directed based on destination."""
    destination: str    # CIDR block (e.g., "0.0.0.0/0" for default route)
    target: str         # e.g., "igw-001" (Internet Gateway), "nat-001", "local"
    description: str = ""


@dataclass
class RouteTable:
    """Routing table associated with a subnet.
    The key difference between public and private subnets is the default route:
    - Public: 0.0.0.0/0 -> Internet Gateway (direct internet access)
    - Private: 0.0.0.0/0 -> NAT Gateway (outbound only, no inbound from internet)
    """
    name: str
    routes: List[RouteEntry] = field(default_factory=list)

    def lookup(self, dest_ip: str) -> Optional[RouteEntry]:
        """Find the most specific matching route (longest prefix match).
        This is how routers work: more specific routes take priority."""
        dest = ipaddress.IPv4Address(dest_ip)
        best_match: Optional[RouteEntry] = None
        best_prefix = -1
        for route in self.routes:
            network = ipaddress.IPv4Network(route.destination, strict=False)
            if dest in network and network.prefixlen > best_prefix:
                best_match = route
                best_prefix = network.prefixlen
        return best_match


@dataclass
class SecurityGroupRule:
    """Security group rules are STATEFUL -- if you allow inbound traffic,
    the response is automatically allowed outbound (and vice versa).
    This simplifies configuration compared to NACLs."""
    direction: str     # "inbound" or "outbound"
    protocol: str      # "tcp", "udp", "icmp", "-1" (all)
    port_range: Tuple[int, int]  # (from_port, to_port)
    source: str        # CIDR or security group ID
    description: str = ""


@dataclass
class NetworkACLRule:
    """NACLs are STATELESS -- you must explicitly allow both inbound AND outbound.
    They also have rule numbers; lower numbers are evaluated first.
    Use NACLs as an extra defense layer beyond security groups."""
    rule_number: int
    direction: str
    action: str        # "ALLOW" or "DENY"
    protocol: str
    port_range: Tuple[int, int]
    source: str
    description: str = ""


@dataclass
class Subnet:
    """A subnet within a VPC, associated with one Availability Zone."""
    name: str
    cidr: CIDRBlock
    subnet_type: SubnetType
    availability_zone: str
    route_table: RouteTable = field(default_factory=lambda: RouteTable("default"))
    nacl_rules: List[NetworkACLRule] = field(default_factory=list)


class VPC:
    """Simulates a complete VPC with subnets, routing, and security."""

    def __init__(self, name: str, cidr: str):
        self.name = name
        self.cidr = CIDRBlock(cidr)
        self.subnets: Dict[str, Subnet] = {}
        self.security_groups: Dict[str, List[SecurityGroupRule]] = {}
        self.internet_gateway: Optional[str] = None
        self.nat_gateway: Optional[str] = None

    def create_subnets(self) -> None:
        """Create a standard 3-tier VPC layout:
        - 2 public subnets (for load balancers, bastion hosts)
        - 2 private subnets (for application servers)
        This mirrors real-world AWS VPC best practices."""
        sub_cidrs = self.cidr.subdivide(new_prefix=24)

        layout = [
            ("public-a",  SubnetType.PUBLIC,  "az-a", sub_cidrs[0]),
            ("public-b",  SubnetType.PUBLIC,  "az-b", sub_cidrs[1]),
            ("private-a", SubnetType.PRIVATE, "az-a", sub_cidrs[2]),
            ("private-b", SubnetType.PRIVATE, "az-b", sub_cidrs[3]),
        ]

        self.internet_gateway = "igw-001"
        self.nat_gateway = "nat-001"

        for name, stype, az, cidr_str in layout:
            rt = RouteTable(f"rt-{name}")
            # All subnets route to the local VPC network first
            rt.routes.append(RouteEntry(self.cidr.cidr, "local", "VPC local traffic"))

            if stype == SubnetType.PUBLIC:
                rt.routes.append(RouteEntry("0.0.0.0/0", self.internet_gateway,
                                            "Internet via IGW"))
            else:
                rt.routes.append(RouteEntry("0.0.0.0/0", self.nat_gateway,
                                            "Outbound via NAT Gateway"))

            subnet = Subnet(name, CIDRBlock(cidr_str), stype, az, rt)
            self.subnets[name] = subnet

    def create_security_groups(self) -> None:
        """Create typical security groups for a web application."""
        # Web tier: allow HTTP/HTTPS from anywhere
        self.security_groups["sg-web"] = [
            SecurityGroupRule("inbound", "tcp", (80, 80), "0.0.0.0/0", "HTTP"),
            SecurityGroupRule("inbound", "tcp", (443, 443), "0.0.0.0/0", "HTTPS"),
            SecurityGroupRule("outbound", "tcp", (0, 65535), "0.0.0.0/0", "All outbound"),
        ]
        # App tier: only accept traffic from the web security group
        self.security_groups["sg-app"] = [
            SecurityGroupRule("inbound", "tcp", (8080, 8080), "sg-web", "From web tier"),
            SecurityGroupRule("outbound", "tcp", (0, 65535), "0.0.0.0/0", "All outbound"),
        ]
        # DB tier: only accept traffic from the app security group
        self.security_groups["sg-db"] = [
            SecurityGroupRule("inbound", "tcp", (5432, 5432), "sg-app", "PostgreSQL from app"),
            SecurityGroupRule("outbound", "tcp", (0, 65535), "sg-app", "Response to app"),
        ]

    def evaluate_security_group(self, sg_name: str, direction: str,
                                protocol: str, port: int, source: str) -> bool:
        """Check if a packet is allowed by security group rules.
        Security groups are deny-by-default: if no rule matches, traffic is DENIED."""
        rules = self.security_groups.get(sg_name, [])
        for rule in rules:
            if rule.direction != direction:
                continue
            if rule.protocol not in (protocol, "-1"):
                continue
            if not (rule.port_range[0] <= port <= rule.port_range[1]):
                continue
            # In reality, source matching involves CIDR containment checks
            if rule.source in ("0.0.0.0/0", source):
                return True
        return False  # Default deny

    def simulate_packet(self, src_subnet: str, dst_ip: str, port: int) -> None:
        """Trace a packet through VPC routing and security."""
        subnet = self.subnets.get(src_subnet)
        if not subnet:
            print(f"  Subnet '{src_subnet}' not found")
            return

        route = subnet.route_table.lookup(dst_ip)
        if route:
            print(f"  Packet from {src_subnet} -> {dst_ip}:{port}")
            print(f"    Route match: {route.destination} via {route.target} "
                  f"({route.description})")
        else:
            print(f"  No route found for {dst_ip}")


def demo_vpc_architecture():
    """Build and display a complete VPC architecture."""
    print("=" * 70)
    print("VPC Architecture: 10.0.0.0/16")
    print("=" * 70)

    vpc = VPC("production-vpc", "10.0.0.0/16")
    vpc.create_subnets()
    vpc.create_security_groups()

    print(f"\n  VPC: {vpc.name} ({vpc.cidr.cidr})")
    print(f"  Total IPs: {vpc.cidr.total_ips:,}")
    print(f"  IGW: {vpc.internet_gateway}")
    print(f"  NAT: {vpc.nat_gateway}")

    print(f"\n  Subnets:")
    for name, subnet in vpc.subnets.items():
        print(f"    {name:>12}: {subnet.cidr.cidr:<18} "
              f"type={subnet.subnet_type.value:<8} az={subnet.availability_zone}  "
              f"usable_ips={subnet.cidr.usable_ips}")

    print(f"\n  Routing Tables:")
    for name, subnet in vpc.subnets.items():
        print(f"    {subnet.route_table.name}:")
        for route in subnet.route_table.routes:
            print(f"      {route.destination:<20} -> {route.target:<12} ({route.description})")

    print(f"\n  Security Groups:")
    for sg_name, rules in vpc.security_groups.items():
        print(f"    {sg_name}:")
        for rule in rules:
            print(f"      {rule.direction:<10} {rule.protocol:<5} "
                  f"port={rule.port_range[0]}-{rule.port_range[1]:<6} "
                  f"source={rule.source:<14} ({rule.description})")
    print()

    # Simulate packet flows
    print("=" * 70)
    print("Packet Flow Simulation")
    print("=" * 70)
    vpc.simulate_packet("public-a", "8.8.8.8", 443)         # Internet-bound
    vpc.simulate_packet("private-a", "8.8.8.8", 443)        # NAT-bound
    vpc.simulate_packet("private-a", "10.0.0.50", 8080)     # VPC internal

    # Security group evaluation
    print(f"\n  Security Group Evaluation:")
    tests = [
        ("sg-web", "inbound", "tcp", 443, "0.0.0.0/0"),
        ("sg-web", "inbound", "tcp", 22, "0.0.0.0/0"),     # SSH not allowed
        ("sg-app", "inbound", "tcp", 8080, "sg-web"),
        ("sg-db",  "inbound", "tcp", 5432, "sg-app"),
        ("sg-db",  "inbound", "tcp", 5432, "0.0.0.0/0"),   # Direct DB access denied
    ]
    for sg, direction, proto, port, source in tests:
        allowed = vpc.evaluate_security_group(sg, direction, proto, port, source)
        status = "ALLOW" if allowed else "DENY"
        print(f"    {sg} {direction} {proto}:{port} from {source:<14} -> {status}")
    print()


def demo_cidr_planning():
    """Show CIDR block planning for a multi-environment setup."""
    print("=" * 70)
    print("CIDR Planning: Multi-Environment VPC Design")
    print("=" * 70)

    # Common pattern: /16 per VPC, non-overlapping for VPC peering
    environments = {
        "production":  "10.0.0.0/16",
        "staging":     "10.1.0.0/16",
        "development": "10.2.0.0/16",
    }

    for env, cidr_str in environments.items():
        cidr = CIDRBlock(cidr_str)
        subnets = cidr.subdivide(new_prefix=24)
        print(f"\n  {env}: {cidr_str} ({cidr.total_ips:,} IPs)")
        print(f"    Available /24 subnets: {len(subnets)} (first 4 shown)")
        for s in subnets[:4]:
            sub_cidr = CIDRBlock(s)
            print(f"      {s:<20} ({sub_cidr.usable_ips} usable IPs)")
    print()


if __name__ == "__main__":
    random.seed(42)
    demo_vpc_architecture()
    demo_cidr_planning()
