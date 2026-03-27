"""
Exercises for Lesson 07: Subnetting Practice
Topic: Networking
Solutions to practice problems from the lesson.
"""
import math


def ip_to_int(ip):
    """Convert dotted-decimal IP to 32-bit integer."""
    octets = [int(o) for o in ip.split(".")]
    return (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]


def int_to_ip(n):
    """Convert 32-bit integer to dotted-decimal IP."""
    return f"{(n >> 24) & 0xFF}.{(n >> 16) & 0xFF}.{(n >> 8) & 0xFF}.{n & 0xFF}"


def cidr_to_mask(prefix_len):
    """Convert CIDR prefix length to subnet mask integer."""
    return (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF


def calculate_subnet(ip_str, prefix):
    """Calculate subnet information for an IP/prefix."""
    ip_int = ip_to_int(ip_str)
    mask_int = cidr_to_mask(prefix)
    network_int = ip_int & mask_int
    host_bits = 32 - prefix
    broadcast_int = network_int | ((1 << host_bits) - 1)

    return {
        "network": int_to_ip(network_int),
        "broadcast": int_to_ip(broadcast_int),
        "first_host": int_to_ip(network_int + 1),
        "last_host": int_to_ip(broadcast_int - 1),
        "usable_hosts": (2 ** host_bits) - 2,
        "mask": int_to_ip(mask_int),
    }


def exercise_1():
    """
    Problem 1: Basic Subnet Calculation
    Calculate network information for:
    a) 192.168.50.100/27
    b) 10.20.30.200/21
    c) 172.31.128.50/18

    Reasoning: The block-size method (256 - mask_octet) is a quick way to
    find subnet boundaries without binary conversion.
    """
    test_cases = [
        ("192.168.50.100", 27),
        ("10.20.30.200", 21),
        ("172.31.128.50", 18),
    ]

    for ip, prefix in test_cases:
        info = calculate_subnet(ip, prefix)
        print(f"\n  {ip}/{prefix}:")
        print(f"    Subnet mask:      {info['mask']}")
        print(f"    Network address:  {info['network']}")
        print(f"    Broadcast:        {info['broadcast']}")
        print(f"    Host range:       {info['first_host']} - {info['last_host']}")
        print(f"    Usable hosts:     {info['usable_hosts']:,}")

        # Show block-size method for the relevant octet
        mask_octets = [int(o) for o in info["mask"].split(".")]
        for i, mo in enumerate(mask_octets):
            if 0 < mo < 255:
                block = 256 - mo
                ip_octet = int(ip.split(".")[i])
                subnet_start = (ip_octet // block) * block
                print(f"    Block size method: 256-{mo}={block}, "
                      f"{ip_octet}//{block}={ip_octet // block} -> start at .{subnet_start}")
                break


def exercise_2():
    """
    Problem 2: Subnet Division
    a) Divide 192.168.100.0/24 into 8 equal subnets.
    b) How many subnets supporting 16,000 hosts each from 10.0.0.0/8?

    Reasoning: For equal division, borrow enough network bits to create the
    required number of subnets. The tradeoff is fewer hosts per subnet.
    """
    # Part a: Divide /24 into 8 subnets
    print("Part a) Divide 192.168.100.0/24 into 8 subnets:")
    num_subnets = 8
    borrowed_bits = math.ceil(math.log2(num_subnets))
    new_prefix = 24 + borrowed_bits
    block_size = 256 >> borrowed_bits  # 256 / 2^borrowed_bits

    print(f"  Need {num_subnets} subnets -> borrow {borrowed_bits} bits -> /{new_prefix}")
    print(f"  Block size: {block_size}")
    print(f"  {'Subnet':<8s} {'Network Address':<25s} {'Host Range':<25s} {'Broadcast':<20s}")
    print(f"  {'-'*78}")

    for i in range(num_subnets):
        net_start = i * block_size
        info = calculate_subnet(f"192.168.100.{net_start}", new_prefix)
        print(f"  {i + 1:<8d} {info['network'] + '/' + str(new_prefix):<25s} "
              f"{info['first_host'] + ' - ' + info['last_host']:<25s} {info['broadcast']:<20s}")

    # Part b: 16,000 host subnets from 10.0.0.0/8
    print(f"\nPart b) Subnets with 16,000 hosts from 10.0.0.0/8:")
    required = 16000
    host_bits = math.ceil(math.log2(required + 2))
    subnet_prefix = 32 - host_bits
    usable = 2 ** host_bits - 2
    num_possible = 2 ** (subnet_prefix - 8)

    print(f"  16,000 hosts -> need {host_bits} host bits (2^{host_bits}-2 = {usable:,})")
    print(f"  CIDR: /{subnet_prefix}")
    print(f"  Subnets from /8 to /{subnet_prefix}: 2^({subnet_prefix}-8) = {num_possible:,}")


def exercise_3():
    """
    Problem 3: VLSM Design
    Divide 172.30.0.0/23 according to requirements:
    LAN A: 120 hosts, LAN B: 60, LAN C: 30, LAN D: 10,
    WAN Link 1: 2, WAN Link 2: 2

    Reasoning: VLSM allocates subnets of different sizes from largest to smallest,
    avoiding address waste. Always start with the largest requirement.
    """
    requirements = [
        ("LAN A", 120),
        ("LAN B", 60),
        ("LAN C", 30),
        ("LAN D", 10),
        ("WAN Link 1", 2),
        ("WAN Link 2", 2),
    ]

    # Sort by required hosts (largest first) for VLSM allocation
    sorted_reqs = sorted(requirements, key=lambda x: x[1], reverse=True)

    print("VLSM Design for 172.30.0.0/23:")
    print(f"  Total address space: {2 ** (32 - 23)} addresses")
    print(f"\n  Allocating from largest to smallest requirement:")

    current_offset = ip_to_int("172.30.0.0")
    total_space_end = ip_to_int("172.30.0.0") + 2 ** (32 - 23)
    results = []

    print(f"\n  {'Network':<12s} {'Prefix':>8s} {'Network Address':<22s} "
          f"{'Host Range':<30s} {'Broadcast':<18s}")
    print(f"  {'-'*90}")

    for name, needed in sorted_reqs:
        host_bits = math.ceil(math.log2(needed + 2))
        prefix = 32 - host_bits
        block_size = 2 ** host_bits

        # Align to block boundary
        if current_offset % block_size != 0:
            current_offset = ((current_offset // block_size) + 1) * block_size

        net_ip = int_to_ip(current_offset)
        info = calculate_subnet(net_ip, prefix)
        results.append((name, prefix, info))

        print(f"  {name:<12s} /{prefix:<7d} {info['network'] + '/' + str(prefix):<22s} "
              f"{info['first_host'] + ' - ' + info['last_host']:<30s} {info['broadcast']:<18s}")

        current_offset += block_size

    remaining = total_space_end - current_offset
    print(f"\n  Remaining addresses: {remaining}")


def exercise_4():
    """
    Problem 4: Comprehensive Network Design
    Network: 10.50.0.0/20 (4096 addresses)
    Requirements with 30% growth:
    - Sales: 500 (-> 650), Technical: 250 (-> 325), Admin: 100 (-> 130)
    - Server farm: 60 (-> 78), DMZ: 20 (-> 26)

    Reasoning: Always plan for growth. Allocate subnets with enough room
    for the projected increase, not just current needs.
    """
    departments = [
        ("Sales", 500, 1.3),
        ("Technical", 250, 1.3),
        ("Admin", 100, 1.3),
        ("Server Farm", 60, 1.3),
        ("DMZ", 20, 1.3),
    ]

    print("Network Design for 10.50.0.0/20:")
    print(f"  Total space: {2 ** (32 - 20)} addresses")
    print(f"  Growth factor: 30% over 3 years\n")

    current_offset = ip_to_int("10.50.0.0")
    total_used = 0

    # Sort by projected need (largest first)
    planned = [(name, current, math.ceil(current * growth))
               for name, current, growth in departments]
    planned.sort(key=lambda x: x[2], reverse=True)

    print(f"  {'Department':<15s} {'Current':>8s} {'Projected':>10s} "
          f"{'Subnet':>8s} {'Capacity':>10s} {'Network Address':<20s}")
    print(f"  {'-'*75}")

    for name, current, projected in planned:
        host_bits = math.ceil(math.log2(projected + 2))
        prefix = 32 - host_bits
        capacity = 2 ** host_bits - 2
        block = 2 ** host_bits

        # Align
        if current_offset % block != 0:
            current_offset = ((current_offset // block) + 1) * block

        net_ip = int_to_ip(current_offset)
        print(f"  {name:<15s} {current:>8d} {projected:>10d} "
              f"{'/' + str(prefix):>8s} {capacity:>10d} {net_ip + '/' + str(prefix):<20s}")

        current_offset += block
        total_used += block

    remaining = 2 ** (32 - 20) - total_used
    print(f"\n  Total allocated: {total_used} addresses")
    print(f"  Remaining for future: {remaining} addresses")
    print(f"  Utilization: {total_used / 2 ** (32 - 20) * 100:.1f}%")


if __name__ == "__main__":
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
