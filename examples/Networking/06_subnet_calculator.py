"""
Subnet Calculator

Demonstrates:
- CIDR notation parsing
- Network/broadcast address calculation
- Subnet mask operations
- VLSM (Variable Length Subnet Masking) planning
- Supernetting (route aggregation)

Theory:
- CIDR notation: IP/prefix (e.g., 192.168.1.0/24)
- Subnet mask: prefix bits set to 1, host bits to 0.
  /24 → 255.255.255.0 → 256 addresses, 254 usable hosts.
- Network address: IP AND mask. Broadcast: IP OR ~mask.
- VLSM: allocate different-sized subnets from a single block.

Adapted from Networking Lesson 06.
"""

import struct
import socket
from dataclasses import dataclass


# Why: Converting IPs to 32-bit integers enables bitwise AND/OR for
# subnet operations (network addr = IP & mask, broadcast = IP | ~mask).
# This is exactly how hardware NICs and routing ASICs perform these checks.
def ip_to_int(ip: str) -> int:
    return struct.unpack("!I", socket.inet_aton(ip))[0]


def int_to_ip(n: int) -> str:
    return socket.inet_ntoa(struct.pack("!I", n))


def prefix_to_mask(prefix: int) -> int:
    return (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF


def mask_to_prefix(mask: int) -> int:
    count = 0
    while mask & (1 << (31 - count)):
        count += 1
    return count


@dataclass
class SubnetInfo:
    network: str
    broadcast: str
    mask: str
    prefix: int
    total_addresses: int
    usable_hosts: int
    first_host: str
    last_host: str

    def __str__(self) -> str:
        return (f"  Network:    {self.network}/{self.prefix}\n"
                f"  Mask:       {self.mask}\n"
                f"  Broadcast:  {self.broadcast}\n"
                f"  Range:      {self.first_host} - {self.last_host}\n"
                f"  Total:      {self.total_addresses} addresses\n"
                f"  Usable:     {self.usable_hosts} hosts")


def calculate_subnet(cidr: str) -> SubnetInfo:
    """Calculate subnet details from CIDR notation."""
    ip_str, prefix_str = cidr.split("/")
    prefix = int(prefix_str)
    ip = ip_to_int(ip_str)
    mask = prefix_to_mask(prefix)

    network = ip & mask
    broadcast = network | (~mask & 0xFFFFFFFF)
    total = 2 ** (32 - prefix)
    # Why: Subtracting 2 accounts for the network address (all-zeros host part)
    # and broadcast address (all-ones host part), neither of which can be
    # assigned to a host. The edge case for /31 and /32 yields 0 usable hosts
    # under traditional rules (though RFC 3021 allows /31 point-to-point links).
    usable = max(0, total - 2)

    first_host = network + 1 if total > 2 else network
    last_host = broadcast - 1 if total > 2 else broadcast

    return SubnetInfo(
        network=int_to_ip(network),
        broadcast=int_to_ip(broadcast),
        mask=int_to_ip(mask),
        prefix=prefix,
        total_addresses=total,
        usable_hosts=usable,
        first_host=int_to_ip(first_host),
        last_host=int_to_ip(last_host),
    )


def is_in_subnet(ip: str, cidr: str) -> bool:
    """Check if an IP address is within a subnet."""
    ip_int = ip_to_int(ip)
    net_str, prefix_str = cidr.split("/")
    prefix = int(prefix_str)
    mask = prefix_to_mask(prefix)
    network = ip_to_int(net_str) & mask
    return (ip_int & mask) == network


def vlsm_plan(base_cidr: str, requirements: list[tuple[str, int]]) -> list[dict]:
    """VLSM planning: allocate subnets from a base network.

    Args:
        base_cidr: Base network (e.g., "192.168.1.0/24")
        requirements: List of (name, hosts_needed) sorted by size (largest first)
    """
    base_ip_str, base_prefix_str = base_cidr.split("/")
    base_network = ip_to_int(base_ip_str) & prefix_to_mask(int(base_prefix_str))
    base_total = 2 ** (32 - int(base_prefix_str))

    # Why: VLSM allocates largest subnets first to avoid fragmentation.
    # A large subnet needs alignment to a power-of-2 boundary; placing it
    # first guarantees the base network address (already aligned) satisfies
    # this constraint without wasting addresses on padding.
    sorted_reqs = sorted(requirements, key=lambda x: -x[1])

    allocations = []
    current = base_network

    for name, hosts_needed in sorted_reqs:
        # Find smallest prefix that fits
        needed = hosts_needed + 2  # Network + broadcast
        bits = 0
        while (1 << bits) < needed:
            bits += 1
        prefix = 32 - bits
        subnet_size = 1 << bits

        # Why: Subnets must start at addresses divisible by their size.
        # A /26 (64 addresses) can only start at 0, 64, 128, 192, etc.
        # Misaligned subnets would overlap with adjacent address blocks.
        if current % subnet_size != 0:
            current = ((current // subnet_size) + 1) * subnet_size

        if current + subnet_size > base_network + base_total:
            allocations.append({
                "name": name,
                "error": f"Not enough space (need {needed}, none left)",
            })
            continue

        info = calculate_subnet(f"{int_to_ip(current)}/{prefix}")
        allocations.append({
            "name": name,
            "hosts_needed": hosts_needed,
            "subnet": f"{int_to_ip(current)}/{prefix}",
            "info": info,
        })
        current += subnet_size

    return allocations


# Why: Supernetting (route aggregation) is the inverse of subnetting — it
# combines multiple contiguous prefixes into one shorter prefix. This reduces
# routing table size, which directly improves lookup speed and memory usage
# in routers. ISPs aggregate customer prefixes before advertising to peers.
def supernet(cidrs: list[str]) -> str | None:
    """Find the smallest supernet that covers all given CIDRs."""
    if not cidrs:
        return None

    networks = []
    for cidr in cidrs:
        ip_str, prefix_str = cidr.split("/")
        prefix = int(prefix_str)
        network = ip_to_int(ip_str) & prefix_to_mask(prefix)
        broadcast = network | (~prefix_to_mask(prefix) & 0xFFFFFFFF)
        networks.append((network, broadcast))

    min_net = min(n[0] for n in networks)
    max_bcast = max(n[1] for n in networks)

    # Why: XOR reveals where the two addresses first differ in their bits.
    # The number of differing bits tells us how many host bits we need,
    # which determines the shortest prefix that spans the entire range.
    diff = min_net ^ max_bcast
    prefix = 32
    while diff > 0:
        diff >>= 1
        prefix -= 1

    super_network = min_net & prefix_to_mask(prefix)
    return f"{int_to_ip(super_network)}/{prefix}"


# ── Demos ──────────────────────────────────────────────────────────────

def demo_subnet_calc():
    print("=" * 60)
    print("SUBNET CALCULATOR")
    print("=" * 60)

    cidrs = ["192.168.1.0/24", "10.0.0.0/8", "172.16.0.0/20",
             "192.168.1.128/25", "10.10.10.0/30"]

    for cidr in cidrs:
        info = calculate_subnet(cidr)
        print(f"\n  {cidr}:")
        print(f"  {info}")


def demo_membership():
    print("\n" + "=" * 60)
    print("SUBNET MEMBERSHIP CHECK")
    print("=" * 60)

    subnet = "192.168.1.0/24"
    test_ips = ["192.168.1.1", "192.168.1.254", "192.168.2.1",
                "192.168.0.255", "10.0.0.1"]

    print(f"\n  Subnet: {subnet}")
    print(f"\n  {'IP Address':<20} {'In Subnet?':>10}")
    print(f"  {'-'*20} {'-'*10}")
    for ip in test_ips:
        result = is_in_subnet(ip, subnet)
        print(f"  {ip:<20} {'Yes' if result else 'No':>10}")


def demo_vlsm():
    print("\n" + "=" * 60)
    print("VLSM PLANNING")
    print("=" * 60)

    base = "192.168.10.0/24"
    requirements = [
        ("Sales", 60),
        ("Engineering", 28),
        ("Management", 12),
        ("Server Room", 4),
        ("Point-to-Point Link", 2),
    ]

    print(f"\n  Base network: {base} (256 addresses)")
    print(f"\n  Requirements:")
    for name, hosts in requirements:
        print(f"    {name}: {hosts} hosts")

    allocations = vlsm_plan(base, requirements)
    print(f"\n  VLSM Allocation:")
    print(f"  {'Name':<22} {'Subnet':<22} {'Hosts':<8} {'Usable':<8}")
    print(f"  {'-'*22} {'-'*22} {'-'*8} {'-'*8}")

    total_used = 0
    for alloc in allocations:
        if "error" in alloc:
            print(f"  {alloc['name']:<22} ERROR: {alloc['error']}")
        else:
            info = alloc["info"]
            print(f"  {alloc['name']:<22} {alloc['subnet']:<22} "
                  f"{alloc['hosts_needed']:<8} {info.usable_hosts:<8}")
            total_used += info.total_addresses

    print(f"\n  Total used: {total_used}/256 addresses "
          f"({total_used/256*100:.1f}%)")


def demo_supernetting():
    print("\n" + "=" * 60)
    print("SUPERNETTING (ROUTE AGGREGATION)")
    print("=" * 60)

    subnets = [
        "192.168.0.0/24",
        "192.168.1.0/24",
        "192.168.2.0/24",
        "192.168.3.0/24",
    ]

    print(f"\n  Individual routes:")
    for s in subnets:
        print(f"    {s}")

    result = supernet(subnets)
    print(f"\n  Aggregated route: {result}")
    if result:
        info = calculate_subnet(result)
        print(f"  Covers: {info.first_host} to {info.last_host}")
        print(f"  Reduces routing table from {len(subnets)} entries to 1")


if __name__ == "__main__":
    demo_subnet_calc()
    demo_membership()
    demo_vlsm()
    demo_supernetting()
