"""
OSI Layer Packet Builder

Demonstrates:
- Building Ethernet, IP, and TCP headers as raw bytes
- Layer encapsulation (data → TCP → IP → Ethernet)
- Header field breakdown and visualization
- Checksum calculation

Theory:
- The OSI model organizes network communication into 7 layers.
- Each layer adds its own header (encapsulation).
- Ethernet (L2): src/dst MAC, EtherType
- IP (L4→actually L3): version, TTL, protocol, src/dst IP
- TCP (L4): src/dst port, sequence number, flags, window

Adapted from Networking Lesson 02.
"""

import struct
import socket


# ── Ethernet Frame ─────────────────────────────────────────────────────

# Why: struct.pack with network byte order ("!") mirrors how NICs actually
# transmit frames on the wire — big-endian, MSB first. Building headers as
# raw bytes (not strings) lets us see exact on-wire representation.
def build_ethernet_header(dst_mac: str, src_mac: str,
                          ethertype: int = 0x0800) -> bytes:
    """Build a 14-byte Ethernet header.

    Args:
        dst_mac: Destination MAC (e.g., "AA:BB:CC:DD:EE:FF")
        src_mac: Source MAC
        ethertype: 0x0800 = IPv4, 0x86DD = IPv6, 0x0806 = ARP
    """
    dst = bytes(int(b, 16) for b in dst_mac.split(":"))
    src = bytes(int(b, 16) for b in src_mac.split(":"))
    return dst + src + struct.pack("!H", ethertype)


def parse_ethernet_header(data: bytes) -> dict:
    dst = ":".join(f"{b:02x}" for b in data[0:6])
    src = ":".join(f"{b:02x}" for b in data[6:12])
    ethertype = struct.unpack("!H", data[12:14])[0]
    return {"dst_mac": dst, "src_mac": src, "ethertype": f"0x{ethertype:04x}"}


# ── IP Header ──────────────────────────────────────────────────────────

# Why: The IP checksum uses one's-complement addition per RFC 1071. We fold
# carry bits back into the 16-bit sum because one's-complement arithmetic
# wraps overflow, unlike two's-complement. This allows routers to verify
# header integrity with minimal computation — a design choice for speed
# at every hop along the path.
def ip_checksum(header: bytes) -> int:
    """Calculate IP header checksum (RFC 1071)."""
    if len(header) % 2 == 1:
        header += b'\x00'
    total = 0
    for i in range(0, len(header), 2):
        total += struct.unpack("!H", header[i:i+2])[0]
    total = (total >> 16) + (total & 0xFFFF)
    total += total >> 16
    return ~total & 0xFFFF


def build_ip_header(src_ip: str, dst_ip: str, payload_len: int,
                    protocol: int = 6, ttl: int = 64) -> bytes:
    """Build a 20-byte IPv4 header (no options).

    Args:
        protocol: 6 = TCP, 17 = UDP, 1 = ICMP
    """
    version_ihl = (4 << 4) | 5  # IPv4, 5 × 4 = 20 bytes
    dscp_ecn = 0
    total_length = 20 + payload_len
    identification = 54321
    flags_offset = 0x4000  # Don't Fragment
    # Why: Checksum is initially set to 0 because the checksum computation
    # includes the checksum field itself. Setting it to 0 first, computing
    # the checksum, then patching it in is the standard two-pass approach.
    header_checksum = 0

    src = socket.inet_aton(src_ip)
    dst = socket.inet_aton(dst_ip)

    header = struct.pack(
        "!BBHHHBBH4s4s",
        version_ihl, dscp_ecn, total_length,
        identification, flags_offset,
        ttl, protocol, header_checksum,
        src, dst,
    )

    checksum = ip_checksum(header)
    header = header[:10] + struct.pack("!H", checksum) + header[12:]
    return header


def parse_ip_header(data: bytes) -> dict:
    fields = struct.unpack("!BBHHHBBH4s4s", data[:20])
    return {
        "version": fields[0] >> 4,
        "ihl": (fields[0] & 0x0F) * 4,
        "total_length": fields[2],
        "ttl": fields[5],
        "protocol": {6: "TCP", 17: "UDP", 1: "ICMP"}.get(fields[6], str(fields[6])),
        "checksum": f"0x{fields[7]:04x}",
        "src_ip": socket.inet_ntoa(fields[8]),
        "dst_ip": socket.inet_ntoa(fields[9]),
    }


# ── TCP Header ─────────────────────────────────────────────────────────

# Why: TCP flags are single bits in the flags byte, so we use a bitmask dict.
# This lets us combine flags via bitwise OR (e.g., SYN+ACK = 0x12), matching
# how real TCP stacks set multiple flags simultaneously in one byte.
TCP_FLAGS = {
    "FIN": 0x01, "SYN": 0x02, "RST": 0x04, "PSH": 0x08,
    "ACK": 0x10, "URG": 0x20,
}


def build_tcp_header(src_port: int, dst_port: int,
                     seq: int = 0, ack: int = 0,
                     flags: list[str] | None = None,
                     window: int = 65535) -> bytes:
    """Build a 20-byte TCP header (no options)."""
    flag_bits = 0
    for f in (flags or []):
        flag_bits |= TCP_FLAGS.get(f, 0)

    data_offset = 5 << 4  # 5 × 4 = 20 bytes, upper nibble
    checksum = 0
    urgent_ptr = 0

    return struct.pack(
        "!HHIIBBHHH",
        src_port, dst_port,
        seq, ack,
        data_offset, flag_bits,
        window,
        checksum, urgent_ptr,
    )


def parse_tcp_header(data: bytes) -> dict:
    fields = struct.unpack("!HHIIBBHHH", data[:20])
    flag_bits = fields[5]
    active_flags = [name for name, bit in TCP_FLAGS.items()
                    if flag_bits & bit]
    return {
        "src_port": fields[0],
        "dst_port": fields[1],
        "seq": fields[2],
        "ack": fields[3],
        "data_offset": (fields[4] >> 4) * 4,
        "flags": active_flags,
        "window": fields[6],
    }


# ── Demos ──────────────────────────────────────────────────────────────

def demo_headers():
    print("=" * 60)
    print("PACKET HEADER CONSTRUCTION")
    print("=" * 60)

    # Build TCP segment
    tcp = build_tcp_header(
        src_port=12345, dst_port=80,
        seq=1000, ack=0,
        flags=["SYN"],
    )
    print(f"\n  TCP Header ({len(tcp)} bytes):")
    tcp_parsed = parse_tcp_header(tcp)
    for k, v in tcp_parsed.items():
        print(f"    {k:<15} {v}")

    # Build IP datagram
    ip = build_ip_header("192.168.1.100", "93.184.216.34",
                         payload_len=len(tcp))
    print(f"\n  IP Header ({len(ip)} bytes):")
    ip_parsed = parse_ip_header(ip)
    for k, v in ip_parsed.items():
        print(f"    {k:<15} {v}")

    # Build Ethernet frame
    eth = build_ethernet_header("AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66")
    print(f"\n  Ethernet Header ({len(eth)} bytes):")
    eth_parsed = parse_ethernet_header(eth)
    for k, v in eth_parsed.items():
        print(f"    {k:<15} {v}")


# Why: This demo shows the encapsulation overhead cost — each layer wraps
# the previous with its own header. Understanding this overhead is critical
# for MTU planning: a 1500-byte Ethernet MTU leaves only ~1460 bytes for
# application data after TCP/IP headers.
def demo_encapsulation():
    print("\n" + "=" * 60)
    print("LAYER ENCAPSULATION")
    print("=" * 60)

    payload = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
    print(f"\n  Application data: {len(payload)} bytes")
    print(f"    {payload[:50]}...")

    tcp = build_tcp_header(12345, 80, seq=1, flags=["PSH", "ACK"])
    tcp_segment = tcp + payload
    print(f"\n  + TCP header: {len(tcp)} bytes → segment: {len(tcp_segment)} bytes")

    ip = build_ip_header("10.0.0.1", "93.184.216.34", len(tcp_segment))
    ip_packet = ip + tcp_segment
    print(f"  + IP header:  {len(ip)} bytes → packet:  {len(ip_packet)} bytes")

    eth = build_ethernet_header("AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66")
    frame = eth + ip_packet
    print(f"  + Eth header: {len(eth)} bytes → frame:   {len(frame)} bytes")

    print(f"\n  Overhead: {len(frame) - len(payload)} bytes "
          f"({(len(frame) - len(payload)) / len(frame) * 100:.1f}%)")

    # Hex dump of complete frame
    print(f"\n  Complete frame (hex):")
    hex_str = frame.hex()
    for i in range(0, len(hex_str), 32):
        offset = i // 2
        chunk = hex_str[i:i+32]
        # Add spaces every 2 chars
        spaced = " ".join(chunk[j:j+2] for j in range(0, len(chunk), 2))
        print(f"    {offset:04x}  {spaced}")


def demo_tcp_flags():
    print("\n" + "=" * 60)
    print("TCP FLAG COMBINATIONS")
    print("=" * 60)

    scenarios = [
        ("SYN (connection request)", ["SYN"]),
        ("SYN+ACK (connection accept)", ["SYN", "ACK"]),
        ("ACK (acknowledgment)", ["ACK"]),
        ("PSH+ACK (data transfer)", ["PSH", "ACK"]),
        ("FIN+ACK (close request)", ["FIN", "ACK"]),
        ("RST (connection reset)", ["RST"]),
    ]

    print(f"\n  {'Scenario':<30} {'Flags':<20} {'Hex':>6}")
    print(f"  {'-'*30} {'-'*20} {'-'*6}")

    for desc, flags in scenarios:
        tcp = build_tcp_header(1234, 80, flags=flags)
        flag_byte = struct.unpack("!B", tcp[13:14])[0]
        print(f"  {desc:<30} {','.join(flags):<20} 0x{flag_byte:02x}")


def demo_checksum():
    print("\n" + "=" * 60)
    print("IP HEADER CHECKSUM")
    print("=" * 60)

    header = build_ip_header("192.168.1.1", "10.0.0.1", 100)
    parsed = parse_ip_header(header)
    print(f"\n  Original checksum: {parsed['checksum']}")

    # Why: Re-computing the checksum over a valid header (including its own
    # checksum field) always yields 0 in one's-complement arithmetic.
    # This is the verification property that routers exploit at every hop.
    verify = ip_checksum(header)
    print(f"  Verification (should be 0): 0x{verify:04x}")

    # Corrupt one byte and re-verify
    corrupted = bytearray(header)
    corrupted[8] ^= 0xFF  # Flip TTL
    verify_bad = ip_checksum(bytes(corrupted))
    print(f"  After corruption: 0x{verify_bad:04x} (non-zero = error detected!)")


if __name__ == "__main__":
    demo_headers()
    demo_encapsulation()
    demo_tcp_flags()
    demo_checksum()
