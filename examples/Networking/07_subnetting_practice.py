"""
Subnetting Practice

Demonstrates:
- Subnet calculation drill problems
- Interactive-style problem generation
- Answer verification
- Common subnetting mistakes

Theory:
- Subnetting divides a network into smaller segments.
- Key skills: calculate network address, broadcast, host range,
  number of usable hosts from CIDR notation.
- Common pitfalls: forgetting network/broadcast addresses,
  miscounting host bits, off-by-one errors.

Adapted from Networking Lesson 07.
"""

import struct
import socket
import random


def ip_to_int(ip: str) -> int:
    return struct.unpack("!I", socket.inet_aton(ip))[0]


def int_to_ip(n: int) -> str:
    return socket.inet_ntoa(struct.pack("!I", n))


def prefix_to_mask(prefix: int) -> int:
    return (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF


def subnet_info(ip: str, prefix: int) -> dict:
    """Calculate all subnet details."""
    ip_int = ip_to_int(ip)
    mask = prefix_to_mask(prefix)
    network = ip_int & mask
    broadcast = network | (~mask & 0xFFFFFFFF)
    total = 2 ** (32 - prefix)
    usable = max(0, total - 2)

    return {
        "network": int_to_ip(network),
        "broadcast": int_to_ip(broadcast),
        "mask": int_to_ip(mask),
        "prefix": prefix,
        "first_host": int_to_ip(network + 1) if total > 2 else int_to_ip(network),
        "last_host": int_to_ip(broadcast - 1) if total > 2 else int_to_ip(broadcast),
        "total": total,
        "usable": usable,
    }


# ── Problem Generator ──────────────────────────────────────────────────

# Why: Using a seeded PRNG ensures reproducible problem sets. This is
# important for practice drills — students can retry the same problems
# to verify their answers, and instructors get consistent test sets.
class SubnettingProblem:
    """Generate and verify subnetting problems."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate(self) -> dict:
        """Generate a random subnetting problem."""
        # Why: Generating from each classful range equally ensures students
        # practice with all address classes. Real-world CIDR is classless, but
        # exam questions still reference classful boundaries (A: 1-126,
        # B: 128-191, C: 192-223), so familiarity with both is valuable.
        first_octet = self.rng.choice([
            self.rng.randint(1, 126),     # Class A
            self.rng.randint(128, 191),   # Class B
            self.rng.randint(192, 223),   # Class C
        ])
        ip = f"{first_octet}.{self.rng.randint(0,255)}.{self.rng.randint(0,255)}.{self.rng.randint(1,254)}"
        prefix = self.rng.randint(8, 30)

        info = subnet_info(ip, prefix)
        return {
            "ip": ip,
            "prefix": prefix,
            "answer": info,
        }

    def generate_host_count(self) -> dict:
        """Generate: 'How many hosts in /N?'"""
        prefix = self.rng.randint(8, 30)
        total = 2 ** (32 - prefix)
        usable = max(0, total - 2)
        return {
            "question": f"How many usable hosts in a /{prefix} network?",
            "answer": usable,
            "total_addresses": total,
        }

    # Why: "Are these two IPs on the same subnet?" is the most practical
    # subnetting skill — network admins must answer this to determine if
    # traffic routes directly (same subnet) or needs a gateway (different subnet).
    def generate_same_subnet(self) -> dict:
        """Generate: 'Are these IPs in the same subnet?'"""
        prefix = self.rng.randint(16, 28)
        mask = prefix_to_mask(prefix)

        # Generate base network
        base = self.rng.randint(0x0A000000, 0x0AFFFFFF)  # 10.x.x.x
        network = base & mask

        # IP 1 always in the subnet
        ip1_int = network + self.rng.randint(1, min(100, 2**(32-prefix) - 2))
        ip1 = int_to_ip(ip1_int)

        # Why: 50/50 split between same/different subnet prevents students from
        # guessing. If most answers were "same," the drill would have no value.
        same = self.rng.choice([True, False])
        if same:
            ip2_int = network + self.rng.randint(1, min(200, 2**(32-prefix) - 2))
        else:
            # Different subnet
            ip2_int = (network + 2**(32-prefix) +
                       self.rng.randint(1, min(100, 2**(32-prefix) - 2)))
        ip2 = int_to_ip(ip2_int & 0xFFFFFFFF)

        # Why: The definitive same-subnet test is (IP1 AND mask) == (IP2 AND mask).
        # We recompute this from the actual IPs rather than trusting the `same`
        # flag because integer overflow could place ip2 in an unexpected subnet.
        actual_same = (ip_to_int(ip1) & mask) == (ip_to_int(ip2) & mask)
        return {
            "ip1": ip1,
            "ip2": ip2,
            "prefix": prefix,
            "same_subnet": actual_same,
        }


# ── Demos ──────────────────────────────────────────────────────────────

def demo_practice_problems():
    print("=" * 60)
    print("SUBNETTING PRACTICE PROBLEMS")
    print("=" * 60)

    gen = SubnettingProblem(seed=42)

    for i in range(5):
        problem = gen.generate()
        ip = problem["ip"]
        prefix = problem["prefix"]
        ans = problem["answer"]

        print(f"\n  Problem {i+1}: Given {ip}/{prefix}")
        print(f"  {'Question':<25} {'Answer'}")
        print(f"  {'-'*25} {'-'*20}")
        print(f"  {'Network address':<25} {ans['network']}")
        print(f"  {'Broadcast address':<25} {ans['broadcast']}")
        print(f"  {'Subnet mask':<25} {ans['mask']}")
        print(f"  {'First usable host':<25} {ans['first_host']}")
        print(f"  {'Last usable host':<25} {ans['last_host']}")
        print(f"  {'Usable hosts':<25} {ans['usable']}")


def demo_host_count():
    print("\n" + "=" * 60)
    print("HOST COUNT REFERENCE")
    print("=" * 60)

    print(f"\n  {'Prefix':>8}  {'Mask':<18}  {'Total':>8}  {'Usable':>8}")
    print(f"  {'-'*8}  {'-'*18}  {'-'*8}  {'-'*8}")

    for prefix in [8, 16, 20, 22, 24, 25, 26, 27, 28, 29, 30]:
        mask = int_to_ip(prefix_to_mask(prefix))
        total = 2 ** (32 - prefix)
        usable = max(0, total - 2)
        print(f"  /{prefix:<7}  {mask:<18}  {total:>8}  {usable:>8}")


def demo_same_subnet():
    print("\n" + "=" * 60)
    print("SAME SUBNET? PROBLEMS")
    print("=" * 60)

    gen = SubnettingProblem(seed=42)

    for i in range(6):
        problem = gen.generate_same_subnet()
        ip1 = problem["ip1"]
        ip2 = problem["ip2"]
        prefix = problem["prefix"]
        same = problem["same_subnet"]

        # Show work
        mask = prefix_to_mask(prefix)
        net1 = int_to_ip(ip_to_int(ip1) & mask)
        net2 = int_to_ip(ip_to_int(ip2) & mask)

        print(f"\n  Problem {i+1}: Are {ip1} and {ip2} in /{prefix}?")
        print(f"    {ip1} AND mask = {net1}")
        print(f"    {ip2} AND mask = {net2}")
        print(f"    Same subnet? {'YES' if same else 'NO'}")


def demo_common_mistakes():
    print("\n" + "=" * 60)
    print("COMMON SUBNETTING MISTAKES")
    print("=" * 60)

    print(f"""
  Mistake 1: Confusing network address with first host
  ─────────────────────────────────────────────────────
  192.168.1.0/24:
    Network address: 192.168.1.0   ← NOT usable as host
    First host:      192.168.1.1   ← First usable
    Last host:       192.168.1.254
    Broadcast:       192.168.1.255 ← NOT usable as host

  Mistake 2: Wrong host count for /31 and /32
  ────────────────────────────────────────────
  /31: 2 addresses, 0 usable hosts (RFC 3021 allows point-to-point)
  /32: 1 address, 0 usable hosts (host route)
  /30: 4 addresses, 2 usable hosts (smallest practical subnet)

  Mistake 3: Forgetting VLSM alignment
  ─────────────────────────────────────
  A /26 subnet (64 addresses) must start at a multiple of 64:
    ✓ 192.168.1.0/26    (0 is multiple of 64)
    ✓ 192.168.1.64/26   (64 is multiple of 64)
    ✗ 192.168.1.50/26   (50 is NOT multiple of 64)""")

    # Demonstrate /31
    print(f"\n  /31 Point-to-Point Link:")
    info = subnet_info("10.0.0.0", 31)
    print(f"    Network:   {info['network']}")
    print(f"    Broadcast: {info['broadcast']}")
    print(f"    Usable:    {info['usable']} (but RFC 3021 uses both)")


if __name__ == "__main__":
    demo_practice_problems()
    demo_host_count()
    demo_same_subnet()
    demo_common_mistakes()
