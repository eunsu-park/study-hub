"""
Exercises for Lesson 15: Network Security Basics
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: Security Basics
    - Explain the three elements of the CIA Triad.
    - What is Defense in Depth?
    """
    cia_triad = {
        "Confidentiality": {
            "definition": "Only authorized parties can access the data",
            "mechanisms": ["Encryption (AES, TLS)", "Access control (RBAC)", "Authentication"],
            "threat": "Data breach, unauthorized access",
        },
        "Integrity": {
            "definition": "Data is not modified by unauthorized parties",
            "mechanisms": ["Hashing (SHA-256)", "Digital signatures", "Checksums"],
            "threat": "Data tampering, man-in-the-middle",
        },
        "Availability": {
            "definition": "Systems and data are accessible when needed",
            "mechanisms": ["Redundancy", "Load balancing", "DDoS protection"],
            "threat": "DoS/DDoS attacks, system failures",
        },
    }

    print("CIA Triad:")
    for element, info in cia_triad.items():
        print(f"\n  {element}:")
        print(f"    Definition: {info['definition']}")
        print(f"    Mechanisms: {', '.join(info['mechanisms'])}")
        print(f"    Threat example: {info['threat']}")

    print("\n\nDefense in Depth:")
    print("  Multiple layers of security controls so that if one fails,")
    print("  others still protect the asset. Like a castle with moat,")
    print("  walls, towers, and guards.")
    layers = [
        ("Physical", "Locked server rooms, biometric access"),
        ("Network", "Firewalls, IDS/IPS, VPN"),
        ("Host", "Antivirus, OS hardening, patching"),
        ("Application", "Input validation, WAF, secure coding"),
        ("Data", "Encryption, access controls, backups"),
    ]
    print("\n  Layers:")
    for layer, example in layers:
        print(f"    {layer:15s}: {example}")


def exercise_2():
    """
    Problem 2: Firewalls
    - Packet filtering vs stateful firewall.
    - Analyze iptables rule.
    """
    print("Packet Filtering vs Stateful Firewall:")

    print("\n  Packet Filtering (Stateless):")
    print("    - Examines each packet independently")
    print("    - Checks: src/dst IP, src/dst port, protocol")
    print("    - No connection tracking")
    print("    - Fast but limited (cannot detect context)")

    print("\n  Stateful Firewall:")
    print("    - Tracks connection state (NEW, ESTABLISHED, RELATED)")
    print("    - Understands connection context")
    print("    - Can allow return traffic for established connections")
    print("    - More secure but uses more memory/CPU")

    print("\n\niptables Rule Analysis:")
    print("  iptables -A INPUT -p tcp --dport 22 -s 192.168.1.0/24 -j ACCEPT")
    print()
    print("  Breakdown:")
    rule_parts = [
        ("-A INPUT", "Append to INPUT chain (incoming traffic)"),
        ("-p tcp", "Match TCP protocol"),
        ("--dport 22", "Match destination port 22 (SSH)"),
        ("-s 192.168.1.0/24", "Match source from 192.168.1.0/24 subnet"),
        ("-j ACCEPT", "Action: allow the packet"),
    ]
    for part, meaning in rule_parts:
        print(f"    {part:25s} -> {meaning}")
    print(f"\n  Effect: Allow SSH connections from the 192.168.1.0/24 network only.")
    print(f"  SSH from any other source will be handled by the default policy.")


def exercise_3():
    """
    Problem 3: NAT
    - Two main purposes of NAT.
    - Operating principle of PAT (Port Address Translation).
    """
    print("NAT (Network Address Translation):")
    print("\n  Two main purposes:")
    print("    1. IP address conservation:")
    print("       Multiple devices share one public IP address")
    print("       Essential for IPv4 address exhaustion mitigation")
    print("    2. Security (hide internal topology):")
    print("       Internal IP addresses not directly reachable from Internet")
    print("       Adds a layer of obscurity (not a substitute for firewall)")

    print("\n\n  PAT (Port Address Translation) / NAT Overload:")
    print("  Principle: Maps (internal_IP:port) -> (public_IP:unique_port)")

    # Simulate PAT table
    pat_table = [
        ("192.168.1.10", 50001, "203.0.113.1", 40001, "8.8.8.8", 443),
        ("192.168.1.20", 50002, "203.0.113.1", 40002, "8.8.8.8", 443),
        ("192.168.1.10", 50003, "203.0.113.1", 40003, "1.1.1.1", 80),
    ]

    print(f"\n  PAT Translation Table:")
    print(f"  {'Internal':>22s} -> {'Translated':>22s} -> {'Destination':>18s}")
    print(f"  {'-'*65}")
    for int_ip, int_port, pub_ip, pub_port, dst_ip, dst_port in pat_table:
        print(f"  {int_ip:>15s}:{int_port:<5d} -> {pub_ip:>15s}:{pub_port:<5d} -> {dst_ip:>12s}:{dst_port}")

    print(f"\n  All three connections share public IP {pat_table[0][2]}")
    print(f"  Distinguished by translated port numbers ({pat_table[0][3]}, {pat_table[1][3]}, {pat_table[2][3]})")


def exercise_4():
    """
    Problem 4: VPN
    - Site-to-Site vs Remote Access VPN.
    - IPsec tunnel mode vs transport mode.
    """
    print("VPN Types:")

    print("\n  Site-to-Site VPN:")
    print("    - Connects two networks (e.g., HQ to branch office)")
    print("    - VPN gateways at each end handle encryption")
    print("    - Transparent to end users")
    print("    - Always-on connection")

    print("\n  Remote Access VPN:")
    print("    - Individual user connects to corporate network")
    print("    - VPN client on user's device")
    print("    - On-demand connection")
    print("    - Used by remote workers, travelers")

    print("\n\nIPsec Modes:")

    print("\n  Tunnel Mode:")
    print("    - Entire original IP packet is encrypted")
    print("    - New IP header added by VPN gateway")
    print("    - Used in Site-to-Site VPN")
    print("    - Original: [IP_A][TCP][Data]")
    print("    - Tunnel:   [New_IP][IPsec][IP_A][TCP][Data]")
    print("                 ^outer^       ^inner (encrypted)^")

    print("\n  Transport Mode:")
    print("    - Only payload is encrypted (IP header preserved)")
    print("    - Used for host-to-host communication")
    print("    - Original: [IP][TCP][Data]")
    print("    - Transport: [IP][IPsec][TCP][Data]")
    print("                      ^encrypted payload^")


def exercise_5():
    """
    Problem 5: Encryption
    - Symmetric vs asymmetric encryption pros/cons.
    - Why TLS uses hybrid encryption.
    """
    print("Symmetric vs Asymmetric Encryption:")
    comparison = [
        ("Speed", "Very fast (AES: ~GB/s)", "Slow (RSA: ~KB/s)"),
        ("Key management", "Hard (shared secret needed)", "Easy (public key distributed freely)"),
        ("Key size", "128-256 bits", "2048-4096 bits (RSA)"),
        ("Use case", "Bulk data encryption", "Key exchange, digital signatures"),
        ("Examples", "AES, ChaCha20", "RSA, ECDH, Ed25519"),
    ]

    print(f"  {'Aspect':18s} {'Symmetric':30s} {'Asymmetric':30s}")
    print(f"  {'-'*78}")
    for aspect, sym, asym in comparison:
        print(f"  {aspect:18s} {sym:30s} {asym:30s}")

    print("\n\nWhy TLS uses Hybrid Encryption:")
    print("  Problem: Symmetric is fast but key exchange is hard.")
    print("  Problem: Asymmetric solves key exchange but is too slow for data.")
    print()
    print("  Solution: Use both!")
    print("    1. Asymmetric (RSA/ECDH): Exchange a session key securely")
    print("    2. Symmetric (AES): Encrypt all data with the session key")
    print("    3. Best of both worlds: Secure key exchange + fast data encryption")


def exercise_6():
    """
    Problem 6: Practical security scenarios.
    """
    scenarios = [
        {
            "scenario": "Remote worker needs to access company internal network",
            "solution": "Remote Access VPN (WireGuard or OpenVPN)",
            "reasoning": "Encrypts traffic over untrusted networks, provides access to internal resources",
        },
        {
            "scenario": "Need to block SQL Injection attacks on web server",
            "solution": "Web Application Firewall (WAF) + parameterized queries",
            "reasoning": "WAF inspects L7 HTTP traffic for injection patterns; "
                         "parameterized queries prevent injection at the code level",
        },
        {
            "scenario": "Secure communication between HQ and branch office",
            "solution": "Site-to-Site VPN (IPsec tunnel mode)",
            "reasoning": "Encrypts all traffic between the two sites; transparent to users; "
                         "always-on connection for seamless access",
        },
    ]

    print("Security Solution Selection:")
    for s in scenarios:
        print(f"\n  Scenario: {s['scenario']}")
        print(f"  Solution: {s['solution']}")
        print(f"  Reasoning: {s['reasoning']}")


def exercise_7():
    """
    Problem 7: Comprehensive Analysis
    Find security vulnerabilities in:
    Internet --- Router --- Internal Network
                  |
               Web Server
    """
    print("Security Vulnerability Analysis:")
    print()
    print("  Network diagram:")
    print("    Internet --- Router --- Internal Network")
    print("                  |")
    print("               Web Server")

    vulnerabilities = [
        ("No DMZ", "Web server is on the same network as internal systems. "
         "If compromised, attacker has direct access to internal network."),
        ("No firewall shown", "Router alone does not provide adequate traffic filtering. "
         "Need dedicated firewall with ACLs."),
        ("Web server exposure", "Web server appears directly connected to router. "
         "Should be in a separate DMZ segment."),
        ("No IDS/IPS", "No intrusion detection to alert on suspicious traffic."),
        ("Single path", "No redundancy. Single router failure takes down everything."),
    ]

    print("\n  Vulnerabilities found:")
    for i, (vuln, detail) in enumerate(vulnerabilities, 1):
        print(f"    {i}. {vuln}: {detail}")

    print("\n  Recommended architecture:")
    print("    Internet -> Firewall -> DMZ (Web Server) -> Internal Firewall -> Internal Net")
    print("    Add IDS/IPS between zones for threat detection.")


def exercise_8():
    """
    Problem 8: Encryption Application
    - How to verify file integrity using hashing.
    - Why hash-only is insufficient for password storage.
    """
    import hashlib

    print("File Integrity Verification with Hashing:")
    print()

    # Simulate file hash verification
    file_content = b"This is the original file content."
    original_hash = hashlib.sha256(file_content).hexdigest()
    print(f"  1. Sender computes hash: SHA-256(file) = {original_hash[:32]}...")
    print(f"  2. Sender shares hash through a trusted channel")
    print(f"  3. Receiver downloads file and computes their own hash")
    print(f"  4. If hashes match: file is intact")
    print(f"  5. If different: file was modified in transit")

    # Tampered file
    tampered = b"This is the tampered file content."
    tampered_hash = hashlib.sha256(tampered).hexdigest()
    print(f"\n  Original hash:  {original_hash[:32]}...")
    print(f"  Tampered hash:  {tampered_hash[:32]}...")
    print(f"  Match: {original_hash == tampered_hash}")

    print("\n\nWhy hash-only is insufficient for passwords:")
    reasons = [
        ("Rainbow tables", "Pre-computed hash tables can reverse common password hashes"),
        ("No salting", "Same password = same hash for all users (reveals duplicates)"),
        ("Fast computation", "SHA-256 is designed to be FAST; attackers can try billions/sec"),
    ]
    for reason, detail in reasons:
        print(f"  - {reason}: {detail}")

    print("\n  Proper password storage:")
    print("    Use: bcrypt, scrypt, or Argon2")
    print("    These add: salt (unique per user) + key stretching (intentionally slow)")
    print("    Example: bcrypt('password', cost=12) -> '$2b$12$...' (unique each time)")


if __name__ == "__main__":
    exercises = [
        exercise_1, exercise_2, exercise_3, exercise_4,
        exercise_5, exercise_6, exercise_7, exercise_8,
    ]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
