"""
Exercises for Lesson 11: UDP and Ports
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: TCP vs UDP Selection
    Choose appropriate protocol for each scenario:
    a) Banking transaction system, b) Live streaming broadcast,
    c) Email transmission, d) Multiplayer game character position sync,
    e) Large file download, f) IoT sensor data collection (every second)
    """
    scenarios = [
        ("Banking transaction system", "TCP",
         "Financial data integrity is non-negotiable; every transaction must arrive correctly"),
        ("Live streaming broadcast", "UDP",
         "Real-time delivery matters more than perfection; buffering ruins the experience"),
        ("Email transmission", "TCP",
         "Email content must arrive complete; SMTP uses TCP for reliable delivery"),
        ("Multiplayer game character position sync", "UDP",
         "Only the latest position matters; retransmitting old positions wastes time"),
        ("Large file download", "TCP",
         "Every byte must arrive correctly; a corrupted file is worthless"),
        ("IoT sensor data (every second)", "UDP",
         "Frequent small messages; missing one reading is acceptable, low overhead preferred"),
    ]

    print("TCP vs UDP Protocol Selection:")
    for scenario, protocol, reason in scenarios:
        print(f"\n  {scenario}")
        print(f"    -> {protocol}: {reason}")


def exercise_2():
    """
    Problem 2: Port Number Matching
    a) HTTPS -> 443, b) MySQL -> 3306, c) SMTP -> 25,
    d) SSH -> 22, e) DNS -> 53
    """
    services = {
        "HTTPS": 443,
        "MySQL": 3306,
        "SMTP": 25,
        "SSH": 22,
        "DNS": 53,
    }

    # Extended list for context
    well_known_ports = {
        20: "FTP Data",
        21: "FTP Control",
        22: "SSH",
        23: "Telnet",
        25: "SMTP",
        53: "DNS",
        67: "DHCP Server",
        68: "DHCP Client",
        80: "HTTP",
        110: "POP3",
        143: "IMAP",
        443: "HTTPS",
        993: "IMAPS",
        995: "POP3S",
        3306: "MySQL",
        5432: "PostgreSQL",
        6379: "Redis",
        8080: "HTTP Alt",
    }

    print("Port Number Matching:")
    for service, port in services.items():
        print(f"  {service:10s} -> Port {port}")

    print("\n  Port number ranges:")
    print("    0-1023:     Well-known ports (require root/admin)")
    print("    1024-49151:  Registered ports (applications)")
    print("    49152-65535: Dynamic/ephemeral ports (client-side)")

    print("\n  Extended reference:")
    for port, service in sorted(well_known_ports.items()):
        print(f"    Port {port:>5d}: {service}")


def exercise_3():
    """
    Problem 3: UDP Header Analysis
    Hex dump: 01 BB 00 35 00 1C 8A 7E

    UDP Header structure (8 bytes total):
    - Source Port: 2 bytes
    - Destination Port: 2 bytes
    - Length: 2 bytes
    - Checksum: 2 bytes

    Reasoning: Understanding binary/hex representation of headers is
    essential for packet analysis with tools like Wireshark or tcpdump.
    """
    hex_dump = "01 BB 00 35 00 1C 8A 7E"
    bytes_list = hex_dump.split()

    # Parse UDP header fields
    src_port = int(bytes_list[0] + bytes_list[1], 16)
    dst_port = int(bytes_list[2] + bytes_list[3], 16)
    length = int(bytes_list[4] + bytes_list[5], 16)
    checksum = bytes_list[6] + bytes_list[7]

    # Map well-known ports to services
    port_services = {22: "SSH", 25: "SMTP", 53: "DNS", 80: "HTTP", 443: "HTTPS"}

    print(f"UDP Header Analysis:")
    print(f"  Hex dump: {hex_dump}")
    print(f"\n  Field breakdown:")
    print(f"    Source Port:      0x{bytes_list[0]}{bytes_list[1]} = {src_port} "
          f"({port_services.get(src_port, 'Unknown')})")
    print(f"    Destination Port: 0x{bytes_list[2]}{bytes_list[3]} = {dst_port} "
          f"({port_services.get(dst_port, 'Unknown')})")
    print(f"    UDP Length:       0x{bytes_list[4]}{bytes_list[5]} = {length} bytes total")
    print(f"    Data Size:        {length} - 8 (header) = {length - 8} bytes")
    print(f"    Checksum:         0x{checksum}")

    print(f"\n  Interpretation: This is a DNS query (port 53)")
    print(f"  sent from an HTTPS client (port 443) carrying {length - 8} bytes of data.")


def exercise_4():
    """
    Problem 4: Socket Identification
    Server handling 3 simultaneous requests:
    Client A: 192.168.1.10:50001 -> Server: 10.0.0.5:80
    Client B: 192.168.1.10:50002 -> Server: 10.0.0.5:80
    Client C: 192.168.1.20:50001 -> Server: 10.0.0.5:80

    Reasoning: TCP uses 5-tuples to uniquely identify connections.
    Even if some fields overlap, the combination must be unique.
    """
    connections = [
        {"name": "Client A", "protocol": "TCP",
         "src_ip": "192.168.1.10", "src_port": 50001,
         "dst_ip": "10.0.0.5", "dst_port": 80},
        {"name": "Client B", "protocol": "TCP",
         "src_ip": "192.168.1.10", "src_port": 50002,
         "dst_ip": "10.0.0.5", "dst_port": 80},
        {"name": "Client C", "protocol": "TCP",
         "src_ip": "192.168.1.20", "src_port": 50001,
         "dst_ip": "10.0.0.5", "dst_port": 80},
    ]

    print("Socket Identification via 5-Tuple:")
    print(f"\n  a) How does the server distinguish these connections?")
    print(f"     Using the 5-tuple: (Protocol, Src IP, Src Port, Dst IP, Dst Port)")
    print(f"     Each combination must be unique.")

    print(f"\n  b) 5-tuple for each connection:")
    for conn in connections:
        five_tuple = (conn["protocol"], conn["src_ip"], conn["src_port"],
                      conn["dst_ip"], conn["dst_port"])
        print(f"     {conn['name']}: {five_tuple}")

    print(f"\n  Uniqueness analysis:")
    print(f"     A vs B: Same Src IP, DIFFERENT Src Port (50001 vs 50002)")
    print(f"     A vs C: DIFFERENT Src IP (192.168.1.10 vs .20), same Src Port")
    print(f"     B vs C: Different in both Src IP AND Src Port")
    print(f"\n  All three are uniquely identified by the 5-tuple.")


if __name__ == "__main__":
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
