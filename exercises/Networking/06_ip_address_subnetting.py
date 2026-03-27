"""
Exercises for Lesson 06: IP Addressing and Subnetting
Topic: Networking
Solutions to practice problems from the lesson.
"""


def ip_to_int(ip):
    """Convert dotted-decimal IP to 32-bit integer."""
    octets = [int(o) for o in ip.split(".")]
    return (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]


def int_to_ip(n):
    """Convert 32-bit integer to dotted-decimal IP."""
    return f"{(n >> 24) & 0xFF}.{(n >> 16) & 0xFF}.{(n >> 8) & 0xFF}.{n & 0xFF}"


def ip_to_binary(ip):
    """Convert IP to binary string representation."""
    return ".".join(f"{int(o):08b}" for o in ip.split("."))


def cidr_to_mask(prefix_len):
    """Convert CIDR prefix length to subnet mask integer."""
    return (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF


def exercise_1():
    """
    Problem: Identify the class of the following IP addresses:
    (a) 10.0.0.1, (b) 172.16.0.1, (c) 192.168.1.1, (d) 224.0.0.1

    Reasoning: Classful addressing (A/B/C/D/E) is historical but still
    tested on exams. The first octet determines the class.
    """
    def classify_ip(ip):
        first_octet = int(ip.split(".")[0])
        if first_octet < 128:
            return "A", "1-126", "/8 default"
        elif first_octet < 192:
            return "B", "128-191", "/16 default"
        elif first_octet < 224:
            return "C", "192-223", "/24 default"
        elif first_octet < 240:
            return "D", "224-239", "Multicast"
        else:
            return "E", "240-255", "Reserved"

    ips = ["10.0.0.1", "172.16.0.1", "192.168.1.1", "224.0.0.1"]

    print("IP Address Class Identification:")
    for ip in ips:
        cls, range_str, note = classify_ip(ip)
        first = int(ip.split(".")[0])
        print(f"  {ip:15s} -> Class {cls} (first octet {first} in {range_str}, {note})")


def exercise_2():
    """
    Problem: What is the CIDR notation for subnet mask 255.255.255.192?

    Reasoning: Converting between dotted-decimal mask and CIDR prefix length
    requires counting the number of consecutive 1-bits in the mask.
    """
    mask = "255.255.255.192"
    binary = ip_to_binary(mask)
    # Count consecutive 1s
    ones = binary.replace(".", "").count("1")

    print(f"Subnet mask: {mask}")
    print(f"Binary: {binary}")
    print(f"Number of 1-bits: {ones}")
    print(f"\nAnswer: /{ones}")
    print(f"  255.255.255.192 = 11111111.11111111.11111111.11000000 = 26 ones")


def exercise_3():
    """
    Problem: For 192.168.1.0/24, find the range and count of usable host IPs.
    """
    network = "192.168.1.0"
    prefix = 24
    host_bits = 32 - prefix
    total_addresses = 2 ** host_bits
    usable_hosts = total_addresses - 2  # Subtract network and broadcast

    net_int = ip_to_int(network)
    first_host = int_to_ip(net_int + 1)
    last_host = int_to_ip(net_int + total_addresses - 2)
    broadcast = int_to_ip(net_int + total_addresses - 1)

    print(f"Network: {network}/{prefix}")
    print(f"  Host bits: {host_bits}")
    print(f"  Total addresses: 2^{host_bits} = {total_addresses}")
    print(f"  Network address: {network}")
    print(f"  First usable: {first_host}")
    print(f"  Last usable: {last_host}")
    print(f"  Broadcast: {broadcast}")
    print(f"  Usable hosts: {usable_hosts}")


def exercise_4():
    """
    Problem: Which is NOT a private IP address?
    (a) 10.255.255.255, (b) 172.32.0.1, (c) 192.168.0.1, (d) 172.16.0.1

    Reasoning: RFC 1918 defines three private address ranges. Any address
    outside these ranges is a public (routable) address.
    """
    private_ranges = [
        ("10.0.0.0", "10.255.255.255", "Class A private (10.0.0.0/8)"),
        ("172.16.0.0", "172.31.255.255", "Class B private (172.16.0.0/12)"),
        ("192.168.0.0", "192.168.255.255", "Class C private (192.168.0.0/16)"),
    ]

    def is_private(ip):
        ip_int = ip_to_int(ip)
        for start, end, _ in private_ranges:
            if ip_to_int(start) <= ip_int <= ip_to_int(end):
                return True
        return False

    test_ips = ["10.255.255.255", "172.32.0.1", "192.168.0.1", "172.16.0.1"]

    print("Private IP Address Check:")
    print(f"  RFC 1918 Private Ranges:")
    for start, end, desc in private_ranges:
        print(f"    {start} - {end} ({desc})")

    print(f"\n  Testing addresses:")
    for ip in test_ips:
        private = is_private(ip)
        marker = "" if private else " <<< NOT PRIVATE (answer)"
        print(f"    {ip:20s} -> {'Private' if private else 'PUBLIC'}{marker}")

    print(f"\n  Answer: (b) 172.32.0.1")
    print(f"  The private range is 172.16.0.0-172.31.255.255; 172.32.x.x is outside this range.")


def exercise_5():
    """
    Problem: Calculate network address, broadcast address, first/last host for
    172.16.50.100/20.
    """
    ip = "172.16.50.100"
    prefix = 20

    mask_int = cidr_to_mask(prefix)
    ip_int = ip_to_int(ip)
    network_int = ip_int & mask_int
    host_bits = 32 - prefix
    broadcast_int = network_int | ((1 << host_bits) - 1)

    network = int_to_ip(network_int)
    broadcast = int_to_ip(broadcast_int)
    first_host = int_to_ip(network_int + 1)
    last_host = int_to_ip(broadcast_int - 1)
    mask = int_to_ip(mask_int)

    print(f"Subnet calculation for {ip}/{prefix}:")
    print(f"  Subnet mask: {mask}")
    print(f"  Network address: {network}")
    print(f"  Broadcast address: {broadcast}")
    print(f"  First host: {first_host}")
    print(f"  Last host: {last_host}")
    print(f"  Usable hosts: {2 ** host_bits - 2}")


def exercise_6():
    """
    Problem: A company has 200 PCs. Choose an appropriate subnet mask and CIDR
    from the 192.168.10.0 network.

    Reasoning: We need at least 200 usable host addresses. The formula is
    2^n - 2 >= 200, where n is the number of host bits.
    """
    import math

    required_hosts = 200
    # Find minimum host bits: 2^n - 2 >= 200
    host_bits = math.ceil(math.log2(required_hosts + 2))
    prefix = 32 - host_bits
    usable = 2 ** host_bits - 2

    mask_int = cidr_to_mask(prefix)
    mask = int_to_ip(mask_int)

    print(f"Subnet for 200 PCs on 192.168.10.0:")
    print(f"  Required hosts: {required_hosts}")
    print(f"  Host bits needed: {host_bits} (2^{host_bits} - 2 = {usable} >= {required_hosts})")
    print(f"  CIDR prefix: /{prefix}")
    print(f"  Subnet mask: {mask}")
    print(f"  Usable host addresses: {usable}")


def exercise_7():
    """
    Problem: When subnetting 10.0.0.0/8 to /16, how many subnets are created?
    """
    original_prefix = 8
    new_prefix = 16
    borrowed_bits = new_prefix - original_prefix
    num_subnets = 2 ** borrowed_bits

    print(f"Subnetting 10.0.0.0/{original_prefix} to /{new_prefix}:")
    print(f"  Borrowed bits: {new_prefix} - {original_prefix} = {borrowed_bits}")
    print(f"  Number of subnets: 2^{borrowed_bits} = {num_subnets}")
    print(f"\n  First few subnets:")
    for i in range(min(5, num_subnets)):
        subnet = f"10.{i}.0.0/{new_prefix}"
        print(f"    {subnet}")
    print(f"    ... (total {num_subnets} subnets)")


def exercise_8():
    """
    Problem: Aggregate (Supernet) the following 4 subnets into one:
    192.168.4.0/24, 192.168.5.0/24, 192.168.6.0/24, 192.168.7.0/24

    Reasoning: Supernetting (CIDR aggregation) combines contiguous subnets
    into a single larger prefix, reducing routing table entries.
    """
    subnets = ["192.168.4.0", "192.168.5.0", "192.168.6.0", "192.168.7.0"]

    print("Supernetting 4 contiguous /24 subnets:")
    for subnet in subnets:
        binary = ip_to_binary(subnet)
        print(f"  {subnet}/24 = {binary}")

    # Find common prefix
    # 192.168.4.0 = 11000000.10101000.00000100.00000000
    # 192.168.5.0 = 11000000.10101000.00000101.00000000
    # 192.168.6.0 = 11000000.10101000.00000110.00000000
    # 192.168.7.0 = 11000000.10101000.00000111.00000000
    # Common prefix: first 22 bits match

    first_int = ip_to_int(subnets[0])
    last_int = ip_to_int(subnets[-1])

    # XOR to find differing bits
    diff = first_int ^ last_int
    # Count bits needed for the range
    import math
    range_bits = math.ceil(math.log2(diff + 1)) if diff > 0 else 0
    new_prefix = 32 - max(range_bits, 32 - 24)  # At least as wide as original /24

    # The aggregate uses the network address of the first subnet
    aggregate_mask = cidr_to_mask(new_prefix)
    aggregate_net = int_to_ip(first_int & aggregate_mask)

    print(f"\n  First 22 bits are identical across all 4 subnets.")
    print(f"  Aggregate: {aggregate_net}/{new_prefix}")
    print(f"  This single route covers all four /24 subnets.")


def exercise_9():
    """
    Problem: Convert IPv6 address 2001:0db8:0000:0000:0000:ff00:0042:8329
    to abbreviated form.
    """
    full_addr = "2001:0db8:0000:0000:0000:ff00:0042:8329"

    # Step 1: Remove leading zeros in each group
    groups = full_addr.split(":")
    shortened = [g.lstrip("0") or "0" for g in groups]
    step1 = ":".join(shortened)

    # Step 2: Replace longest consecutive group of zeros with ::
    step2 = step1.replace(":0:0:0:", "::", 1)

    print(f"IPv6 Address Abbreviation:")
    print(f"  Full:     {full_addr}")
    print(f"  Step 1:   {step1} (remove leading zeros)")
    print(f"  Step 2:   {step2} (replace consecutive zeros with ::)")
    print(f"\n  Answer: 2001:db8::ff00:42:8329")


def exercise_10():
    """
    Problem: Explain why NAT is needed and the advantages/disadvantages.

    Reasoning: NAT was a critical stopgap for IPv4 address exhaustion,
    but it breaks the end-to-end principle and adds complexity.
    """
    print("Network Address Translation (NAT):")
    print(f"\n  Why needed:")
    print(f"    - IPv4 address exhaustion (only ~4.3 billion addresses)")
    print(f"    - Allows many devices to share one public IP")
    print(f"    - Enables private networks to access the Internet")

    advantages = [
        "IP address conservation (many devices share one public IP)",
        "Hide internal network topology (security by obscurity)",
        "No need to renumber when changing ISPs (internal stays same)",
    ]

    disadvantages = [
        "Breaks end-to-end connectivity (peer-to-peer applications suffer)",
        "Protocol compatibility issues (some protocols embed IP in payload)",
        "Adds latency and processing overhead on router",
        "Makes inbound connections difficult without port forwarding",
        "Complicates logging and traceability",
    ]

    print(f"\n  Advantages:")
    for adv in advantages:
        print(f"    + {adv}")

    print(f"\n  Disadvantages:")
    for dis in disadvantages:
        print(f"    - {dis}")

    print(f"\n  IPv6 eliminates the need for NAT with 2^128 addresses,")
    print(f"  restoring true end-to-end connectivity.")


if __name__ == "__main__":
    exercises = [
        exercise_1, exercise_2, exercise_3, exercise_4, exercise_5,
        exercise_6, exercise_7, exercise_8, exercise_9, exercise_10,
    ]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
