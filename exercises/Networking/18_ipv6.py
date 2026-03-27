"""
Exercises for Lesson 18: IPv6
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: Address Shortening
    Shorten IPv6 addresses to shortest form:
    a) 2001:0db8:0000:0000:0000:0000:0000:0001
    b) fe80:0000:0000:0000:0202:b3ff:fe1e:8329
    c) 2001:0db8:0001:0000:0000:0000:0000:0000
    d) 0000:0000:0000:0000:0000:0000:0000:0000
    """
    def shorten_ipv6(full_addr):
        """Shorten an IPv6 address to its canonical short form."""
        # Step 1: Remove leading zeros in each group
        groups = full_addr.split(":")
        shortened = [g.lstrip("0") or "0" for g in groups]

        # Step 2: Find longest consecutive run of "0" groups
        best_start, best_len = -1, 0
        curr_start, curr_len = -1, 0
        for i, g in enumerate(shortened):
            if g == "0":
                if curr_start == -1:
                    curr_start = i
                curr_len += 1
                if curr_len > best_len:
                    best_start, best_len = curr_start, curr_len
            else:
                curr_start, curr_len = -1, 0

        # Step 3: Replace longest run with ::
        if best_len >= 2:
            before = ":".join(shortened[:best_start])
            after = ":".join(shortened[best_start + best_len:])
            result = before + "::" + after
            # Handle edge cases
            if result.startswith(":::"):
                result = "::" + result[3:]
            if result.endswith(":::"):
                result = result[:-3] + "::"
        else:
            result = ":".join(shortened)

        return result

    test_cases = [
        ("2001:0db8:0000:0000:0000:0000:0000:0001", "2001:db8::1"),
        ("fe80:0000:0000:0000:0202:b3ff:fe1e:8329", "fe80::202:b3ff:fe1e:8329"),
        ("2001:0db8:0001:0000:0000:0000:0000:0000", "2001:db8:1::"),
        ("0000:0000:0000:0000:0000:0000:0000:0000", "::"),
    ]

    print("IPv6 Address Shortening:")
    for full, expected in test_cases:
        result = shorten_ipv6(full)
        status = "OK" if result == expected else f"EXPECTED {expected}"
        print(f"\n  Full:     {full}")
        print(f"  Short:    {result} ({status})")

    print("\n  Rules:")
    print("    1. Remove leading zeros in each group: 0db8 -> db8")
    print("    2. Replace longest consecutive all-zero groups with ::")
    print("    3. :: can only appear once in an address")


def exercise_2():
    """
    Problem 2: EUI-64 Conversion
    Given MAC: 00:50:56:A1:B2:C3
    Calculate: EUI-64 identifier, link-local, and global address.
    """
    mac = "00:50:56:A1:B2:C3"
    mac_bytes = mac.split(":")

    print(f"EUI-64 Conversion from MAC: {mac}")
    print()

    # Step 1: Split MAC into two halves
    first_half = mac_bytes[:3]
    second_half = mac_bytes[3:]
    print(f"  Step 1: Split MAC into halves")
    print(f"    First:  {':'.join(first_half)}")
    print(f"    Second: {':'.join(second_half)}")

    # Step 2: Insert FF:FE in the middle
    eui64_bytes = first_half + ["FF", "FE"] + second_half
    print(f"\n  Step 2: Insert FF:FE in middle")
    print(f"    {':'.join(eui64_bytes)}")

    # Step 3: Flip the 7th bit (Universal/Local bit) of the first byte
    first_byte = int(first_half[0], 16)
    flipped_byte = first_byte ^ 0x02  # Toggle bit 1 (0-indexed from right)
    eui64_bytes[0] = f"{flipped_byte:02X}"
    print(f"\n  Step 3: Flip 7th bit (U/L bit) of first byte")
    print(f"    {first_byte:02X} (binary: {first_byte:08b})")
    print(f"    XOR 02:      {0x02:08b}")
    print(f"    Result: {flipped_byte:02X}  (binary: {flipped_byte:08b})")

    # Format as IPv6 interface identifier
    iid = f"{eui64_bytes[0]}{eui64_bytes[1]}:{eui64_bytes[2]}{eui64_bytes[3]}" \
          f":{eui64_bytes[4]}{eui64_bytes[5]}:{eui64_bytes[6]}{eui64_bytes[7]}"
    iid_lower = iid.lower()

    print(f"\n  Interface Identifier (IID): {iid_lower}")
    print(f"  Link-local: fe80::{iid_lower}")
    print(f"  Global (prefix 2001:db8:1::/64): 2001:db8:1::{iid_lower}")


def exercise_3():
    """
    Problem 3: Subnetting
    Allocated: 2001:db8:abcd::/48
    Design for 4 regional offices with 256 subnets each.
    Each subnet supports /64 for end-users.
    """
    print("IPv6 Subnetting Design:")
    print(f"  Allocated prefix: 2001:db8:abcd::/48")
    print(f"  Available bits for subnetting: bits 49-64 (16 bits)")
    print(f"  Total /64 subnets: 2^16 = 65,536")

    print(f"\n  Design: Use bits 49-52 for regions (4 bits)")
    print(f"  Each region gets a /52 (4096 /64 subnets)")

    regions = [
        (0, "Region 0 (Americas)", "2001:db8:abcd:0000::/52"),
        (1, "Region 1 (Europe)", "2001:db8:abcd:1000::/52"),
        (2, "Region 2 (Asia)", "2001:db8:abcd:2000::/52"),
        (3, "Region 3 (Australia)", "2001:db8:abcd:3000::/52"),
    ]

    print(f"\n  Region allocation:")
    for idx, name, prefix in regions:
        subnets_per_region = 2 ** 12  # /52 to /64 = 12 bits
        print(f"    {name}: {prefix}")
        print(f"      Contains {subnets_per_region:,} /64 subnets")

    print(f"\n  Example subnets in Region 0:")
    for i in range(3):
        print(f"    2001:db8:abcd:{i:04x}::/64")
    print(f"    ...")
    print(f"    2001:db8:abcd:0fff::/64")

    print(f"\n  Remaining regions (4-15) reserved for future expansion.")
    print(f"  Each region has far more than 256 subnets (4096 available).")


def exercise_4():
    """
    Problem 4: NDP Analysis
    Explain NDP messages when:
    1. A host boots up and gets IPv6 connectivity.
    2. Host A wants to send to Host B on the same link.
    """
    print("NDP (Neighbor Discovery Protocol) Message Sequences:")

    print("\n  1. Host boot-up sequence:")
    boot_steps = [
        ("Generate link-local", "Create fe80::<IID> from MAC (EUI-64 or random)"),
        ("DAD - Send NS", "Neighbor Solicitation to own solicited-node multicast"),
        ("DAD - Wait", "If no NA received, address is unique"),
        ("Send RS", "Router Solicitation to ff02::2 (all-routers multicast)"),
        ("Receive RA", "Router Advertisement with prefix, M/O flags, router lifetime"),
        ("Generate global", "Combine RA prefix + IID = global unicast address"),
        ("DAD for global", "Repeat DAD for the global address"),
        ("Optional DHCPv6", "If M=1 (managed) or O=1 (other config) in RA"),
    ]
    for step_name, detail in boot_steps:
        print(f"    {step_name:20s}: {detail}")

    print("\n  2. Host A -> Host B on same link:")
    comm_steps = [
        ("Check cache", "A checks neighbor cache for B's link-layer address"),
        ("NS multicast", "If not found, A sends NS to B's solicited-node multicast"),
        ("B receives NS", "B recognizes its own solicited-node multicast"),
        ("B sends NA", "B sends Neighbor Advertisement with its MAC address"),
        ("A caches", "A creates neighbor cache entry (REACHABLE state)"),
        ("Data sent", "A sends packet to B using the learned MAC address"),
    ]
    for step_name, detail in comm_steps:
        print(f"    {step_name:20s}: {detail}")

    print("\n  NDP message types:")
    messages = {
        "RS (Type 133)": "Router Solicitation - host asks for routers",
        "RA (Type 134)": "Router Advertisement - router announces itself",
        "NS (Type 135)": "Neighbor Solicitation - resolve address / DAD",
        "NA (Type 136)": "Neighbor Advertisement - response to NS",
        "Redirect (137)": "Router suggests better next hop",
    }
    for msg, purpose in messages.items():
        print(f"    {msg:20s}: {purpose}")


def exercise_5():
    """
    Problem 5: Transition Mechanism Selection
    a) Enterprise: dual-stack routers, IPv6 islands over IPv4 backbone
    b) Home user: behind NAT, IPv6-only services
    c) ISP: IPv6-only customers, IPv4 content
    d) Small office: IPv4-only ISP, wants IPv6
    """
    scenarios = [
        {
            "scenario": "Enterprise with dual-stack routers, IPv6 islands over IPv4 backbone",
            "mechanism": "6to4 or Manual tunnels (GRE/IPsec)",
            "reasoning": "Controlled environment allows static tunnel configuration. "
                         "6to4 provides automatic tunneling between dual-stack sites.",
        },
        {
            "scenario": "Home user behind NAT with IPv6-only services",
            "mechanism": "Teredo",
            "reasoning": "Teredo tunnels IPv6 through IPv4 NAT without any "
                         "configuration. Automatic, works behind most NATs.",
        },
        {
            "scenario": "ISP serving IPv6-only customers accessing IPv4 content",
            "mechanism": "NAT64 / DNS64",
            "reasoning": "NAT64 translates IPv6 packets to IPv4 at the ISP edge. "
                         "DNS64 synthesizes AAAA records for IPv4-only hosts.",
        },
        {
            "scenario": "Small office, IPv4-only ISP, wants IPv6",
            "mechanism": "6to4 or Tunnel Broker (e.g., Hurricane Electric)",
            "reasoning": "Tunnel broker provides a /48 IPv6 prefix over the "
                         "existing IPv4 connection. Semi-automatic setup.",
        },
    ]

    print("IPv6 Transition Mechanism Selection:")
    for s in scenarios:
        print(f"\n  Scenario: {s['scenario']}")
        print(f"  Recommended: {s['mechanism']}")
        print(f"  Reasoning: {s['reasoning']}")


def exercise_6():
    """
    Problem 6: Security Configuration
    Write ip6tables rules for:
    - Allow established connections
    - Allow essential ICMPv6
    - Allow SSH from specific prefix
    - Drop all other incoming traffic
    """
    rules = [
        ("# Default policies", [
            "ip6tables -P INPUT DROP",
            "ip6tables -P FORWARD DROP",
            "ip6tables -P OUTPUT ACCEPT",
        ]),
        ("# Allow loopback", [
            "ip6tables -A INPUT -i lo -j ACCEPT",
        ]),
        ("# Allow established/related connections", [
            "ip6tables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT",
        ]),
        ("# Allow essential ICMPv6 (error messages)", [
            "ip6tables -A INPUT -p icmpv6 --icmpv6-type destination-unreachable -j ACCEPT",
            "ip6tables -A INPUT -p icmpv6 --icmpv6-type packet-too-big -j ACCEPT",
            "ip6tables -A INPUT -p icmpv6 --icmpv6-type time-exceeded -j ACCEPT",
            "ip6tables -A INPUT -p icmpv6 --icmpv6-type parameter-problem -j ACCEPT",
        ]),
        ("# Allow NDP (essential for IPv6 operation)", [
            "ip6tables -A INPUT -p icmpv6 --icmpv6-type router-advertisement -j ACCEPT",
            "ip6tables -A INPUT -p icmpv6 --icmpv6-type router-solicitation -j ACCEPT",
            "ip6tables -A INPUT -p icmpv6 --icmpv6-type neighbour-solicitation -j ACCEPT",
            "ip6tables -A INPUT -p icmpv6 --icmpv6-type neighbour-advertisement -j ACCEPT",
        ]),
        ("# Allow SSH from 2001:db8::/32 only", [
            "ip6tables -A INPUT -p tcp --dport 22 -s 2001:db8::/32 -j ACCEPT",
        ]),
        ("# Log dropped packets (optional, for debugging)", [
            "ip6tables -A INPUT -j LOG --log-prefix 'IPv6-DROP: '",
        ]),
    ]

    print("IPv6 Firewall Rules (ip6tables):")
    for comment, cmds in rules:
        print(f"\n  {comment}")
        for cmd in cmds:
            print(f"    {cmd}")

    print("\n  Key points:")
    print("    - NDP rules are CRITICAL: blocking them breaks IPv6 entirely")
    print("    - ICMPv6 error messages must be allowed (unlike IPv4 where")
    print("      blocking ICMP is sometimes acceptable)")
    print("    - Path MTU Discovery relies on 'packet-too-big' messages")
    print("    - No NAT in IPv6 means firewall rules are even more important")


if __name__ == "__main__":
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4, exercise_5, exercise_6]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
