"""
IPv6 Demonstration

Demonstrates:
- IPv6 address representation and compression
- IPv6 address types (link-local, global, multicast)
- SLAAC (Stateless Address Autoconfiguration)
- EUI-64 interface ID from MAC address
- IPv4-mapped IPv6 addresses

Theory:
- IPv6 uses 128-bit addresses (vs IPv4's 32 bits).
  Notation: 8 groups of 4 hex digits, separated by colons.
- Compression: leading zeros omitted, one :: for consecutive zero groups.
- SLAAC: host generates its own address from network prefix + MAC address.
  EUI-64: insert FF:FE in middle of MAC, flip 7th bit.
- Address types:
  - ::1 = loopback
  - fe80::/10 = link-local
  - 2000::/3 = global unicast
  - ff00::/8 = multicast

Adapted from Networking Lesson 18.
"""


# Why: IPv6 addresses have two canonical forms — expanded (for parsing and
# comparison) and compressed (for human readability). Converting between them
# is fundamental because the same address can look very different in each form
# (e.g., "::1" vs "0000:0000:0000:0000:0000:0000:0000:0001").
def expand_ipv6(addr: str) -> str:
    """Expand a compressed IPv6 address to full form."""
    if "::" in addr:
        parts = addr.split("::")
        left = parts[0].split(":") if parts[0] else []
        right = parts[1].split(":") if parts[1] else []
        missing = 8 - len(left) - len(right)
        groups = left + ["0000"] * missing + right
    else:
        groups = addr.split(":")

    return ":".join(g.zfill(4) for g in groups)


def compress_ipv6(addr: str) -> str:
    """Compress an IPv6 address (remove leading zeros, use ::)."""
    full = expand_ipv6(addr)
    groups = [g.lstrip("0") or "0" for g in full.split(":")]

    # Why: RFC 5952 requires that :: replaces the LONGEST run of consecutive
    # all-zero groups (and only one :: is allowed). If there are ties, the
    # first run is chosen. This ensures a unique canonical compressed form
    # for every address, preventing ambiguity in log files and configs.
    best_start, best_len = -1, 0
    curr_start, curr_len = -1, 0

    for i, g in enumerate(groups):
        if g == "0":
            if curr_start == -1:
                curr_start = i
            curr_len += 1
            if curr_len > best_len:
                best_start = curr_start
                best_len = curr_len
        else:
            curr_start = -1
            curr_len = 0

    if best_len >= 2:
        left = ":".join(groups[:best_start])
        right = ":".join(groups[best_start + best_len:])
        return f"{left}::{right}" if left and right else \
               f"::{right}" if right else \
               f"{left}::" if left else "::"
    return ":".join(groups)


def ipv6_to_int(addr: str) -> int:
    """Convert IPv6 address to 128-bit integer."""
    full = expand_ipv6(addr)
    return int(full.replace(":", ""), 16)


def int_to_ipv6(n: int) -> str:
    """Convert 128-bit integer to IPv6 address."""
    hex_str = f"{n:032x}"
    groups = [hex_str[i:i+4] for i in range(0, 32, 4)]
    return compress_ipv6(":".join(groups))


# Why: EUI-64 transforms a 48-bit MAC into a 64-bit interface ID for SLAAC.
# Inserting FF:FE pads the MAC to 64 bits, and flipping the U/L bit (7th bit)
# follows IEEE convention — a globally unique MAC gets the bit set to 1 in
# the IPv6 interface ID. This lets hosts auto-generate addresses without DHCP.
def mac_to_eui64(mac: str) -> str:
    """Generate EUI-64 interface ID from MAC address.

    Steps:
    1. Split MAC into two halves
    2. Insert FF:FE in the middle
    3. Flip the 7th bit (Universal/Local bit)
    """
    # Remove separators
    mac_clean = mac.replace(":", "").replace("-", "").lower()
    if len(mac_clean) != 12:
        raise ValueError(f"Invalid MAC address: {mac}")

    # Insert FF:FE
    eui64 = mac_clean[:6] + "fffe" + mac_clean[6:]

    # Flip 7th bit (Universal/Local)
    first_byte = int(eui64[:2], 16) ^ 0x02
    eui64 = f"{first_byte:02x}" + eui64[2:]

    # Format as IPv6 groups
    return ":".join(eui64[i:i+4] for i in range(0, 16, 4))


# Why: SLAAC avoids the need for a DHCPv6 server — the router advertises
# the /64 prefix via Router Advertisement, and the host generates the lower
# 64 bits from its own MAC. This is simpler to deploy but has a privacy
# trade-off: the MAC is embedded in the address (mitigated by RFC 4941
# privacy extensions, which generate random interface IDs instead).
def slaac_address(prefix: str, mac: str) -> str:
    """Generate SLAAC address from network prefix and MAC."""
    eui64 = mac_to_eui64(mac)
    # Combine prefix (first 64 bits) with EUI-64 (last 64 bits)
    prefix_expanded = expand_ipv6(prefix + "::").split(":")
    eui64_groups = eui64.split(":")

    full_groups = prefix_expanded[:4] + eui64_groups
    full_addr = ":".join(full_groups)
    return compress_ipv6(full_addr)


# Why: IPv4-mapped addresses (::ffff:a.b.c.d) are a dual-stack transition
# mechanism. An IPv6 socket can receive IPv4 connections represented as these
# mapped addresses, so a single socket can serve both protocols — essential
# during the long IPv4-to-IPv6 migration.
def ipv4_mapped(ipv4: str) -> str:
    """Create IPv4-mapped IPv6 address."""
    octets = ipv4.split(".")
    hex1 = f"{int(octets[0]):02x}{int(octets[1]):02x}"
    hex2 = f"{int(octets[2]):02x}{int(octets[3]):02x}"
    return f"::ffff:{hex1}:{hex2}"


# Why: IPv6 address classification is based on well-known prefix ranges
# defined by IANA. Unlike IPv4 where classful addressing is obsolete, IPv6
# prefix-based classification is actively used: fe80:: is always link-local,
# 2000::/3 is always global unicast, ff00::/8 is always multicast.
def classify_ipv6(addr: str) -> str:
    """Classify an IPv6 address type."""
    n = ipv6_to_int(addr)
    full = expand_ipv6(addr)

    if n == 1:
        return "Loopback (::1)"
    if n == 0:
        return "Unspecified (::)"
    if full.startswith("fe80"):
        return "Link-Local (fe80::/10)"
    if full.startswith("fc") or full.startswith("fd"):
        return "Unique Local (fc00::/7)"
    if full[:4] >= "2000" and full[:4] <= "3fff":
        return "Global Unicast (2000::/3)"
    if full.startswith("ff"):
        return "Multicast (ff00::/8)"
    if full.startswith("0000:0000:0000:0000:0000:ffff"):
        return "IPv4-Mapped (::ffff:0:0/96)"
    return "Reserved/Unknown"


# ── Demos ──────────────────────────────────────────────────────────────

def demo_compression():
    print("=" * 60)
    print("IPv6 ADDRESS COMPRESSION")
    print("=" * 60)

    addresses = [
        "2001:0db8:0000:0000:0000:0000:0000:0001",
        "fe80:0000:0000:0000:0211:22ff:fe33:4455",
        "2001:0db8:0000:0085:0000:0000:0000:1000",
        "0000:0000:0000:0000:0000:0000:0000:0001",
        "0000:0000:0000:0000:0000:0000:0000:0000",
    ]

    print(f"\n  {'Full Form':<42} {'Compressed'}")
    print(f"  {'-'*42} {'-'*25}")
    for addr in addresses:
        compressed = compress_ipv6(addr)
        print(f"  {addr}  {compressed}")

    # Roundtrip
    print(f"\n  Expansion roundtrip:")
    for addr in ["2001:db8::1", "::1", "fe80::1", "::"]:
        expanded = expand_ipv6(addr)
        recompressed = compress_ipv6(expanded)
        print(f"    {addr:<20} → {expanded} → {recompressed}")


def demo_address_types():
    print("\n" + "=" * 60)
    print("IPv6 ADDRESS TYPES")
    print("=" * 60)

    addresses = [
        "::1",
        "::",
        "fe80::1",
        "fd00::1234:5678",
        "2001:db8::1",
        "2607:f8b0:4004:800::200e",
        "ff02::1",
        "ff02::2",
        "::ffff:c0a8:0101",
    ]

    print(f"\n  {'Address':<30} {'Type'}")
    print(f"  {'-'*30} {'-'*30}")
    for addr in addresses:
        addr_type = classify_ipv6(addr)
        print(f"  {addr:<30} {addr_type}")


def demo_slaac():
    print("\n" + "=" * 60)
    print("SLAAC: STATELESS ADDRESS AUTOCONFIGURATION")
    print("=" * 60)

    mac = "00:11:22:33:44:55"
    prefix = "2001:db8:abcd:1234"

    print(f"\n  MAC Address: {mac}")
    print(f"  Network Prefix: {prefix}::/64")

    eui64 = mac_to_eui64(mac)
    print(f"\n  EUI-64 Generation:")
    print(f"    1. MAC:          {mac}")
    mac_clean = mac.replace(":", "")
    print(f"    2. Split:        {mac_clean[:6]} | {mac_clean[6:]}")
    print(f"    3. Insert FFFE:  {mac_clean[:6]}fffe{mac_clean[6:]}")
    first_byte = int(mac_clean[:2], 16) ^ 0x02
    print(f"    4. Flip 7th bit: {first_byte:02x} "
          f"(0x{int(mac_clean[:2], 16):02x} XOR 0x02)")
    print(f"    5. EUI-64:       {eui64}")

    address = slaac_address(prefix, mac)
    print(f"\n  SLAAC Address: {address}")

    # Multiple MACs
    print(f"\n  SLAAC for different devices on {prefix}::/64:")
    macs = [
        ("Laptop", "AA:BB:CC:DD:EE:FF"),
        ("Phone", "12:34:56:78:9A:BC"),
        ("IoT", "00:1A:2B:3C:4D:5E"),
    ]
    for device, m in macs:
        addr = slaac_address(prefix, m)
        print(f"    {device:<10} {m}  →  {addr}")


def demo_ipv4_mapping():
    print("\n" + "=" * 60)
    print("IPv4-MAPPED IPv6 ADDRESSES")
    print("=" * 60)

    ipv4_addrs = ["192.168.1.1", "10.0.0.1", "8.8.8.8", "172.16.0.1"]

    print(f"\n  {'IPv4':<18} {'IPv4-Mapped IPv6'}")
    print(f"  {'-'*18} {'-'*30}")
    for ipv4 in ipv4_addrs:
        mapped = ipv4_mapped(ipv4)
        print(f"  {ipv4:<18} {mapped}")

    print(f"""
  IPv4-mapped addresses allow IPv6-only applications to
  communicate with IPv4 hosts through a dual-stack socket.
  Format: ::ffff:a.b.c.d""")


if __name__ == "__main__":
    demo_compression()
    demo_address_types()
    demo_slaac()
    demo_ipv4_mapping()
