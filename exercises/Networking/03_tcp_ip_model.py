"""
Exercises for Lesson 03: TCP/IP Model
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem: List the 4 layers of the TCP/IP model from bottom to top.
    """
    layers = [
        (1, "Network Access", "Physical transmission + framing (OSI L1+L2)"),
        (2, "Internet", "IP addressing + routing (OSI L3)"),
        (3, "Transport", "End-to-end delivery - TCP/UDP (OSI L4)"),
        (4, "Application", "User-facing protocols (OSI L5+L6+L7)"),
    ]

    print("TCP/IP 4 Layers (bottom to top):")
    for num, name, description in layers:
        print(f"  Layer {num}: {name:20s} - {description}")


def exercise_2():
    """
    Problem: Classify the following protocols as TCP or UDP:
    HTTP, DNS, FTP, VoIP, SMTP, online gaming
    """
    protocols = {
        "HTTP":          {"transport": "TCP", "reason": "Reliable web page delivery required"},
        "DNS":           {"transport": "UDP (mostly)", "reason": "Small queries, speed important; uses TCP for zone transfers"},
        "FTP":           {"transport": "TCP", "reason": "File integrity requires reliable delivery"},
        "VoIP":          {"transport": "UDP", "reason": "Real-time audio; latency more important than perfect delivery"},
        "SMTP":          {"transport": "TCP", "reason": "Email must be delivered completely and reliably"},
        "Online Gaming": {"transport": "UDP", "reason": "Low latency critical; stale game state data is useless"},
    }

    print("Protocol classification (TCP vs UDP):")
    for proto, info in protocols.items():
        print(f"  {proto:15s} -> {info['transport']:15s} ({info['reason']})")


def exercise_3():
    """
    Problem: Choose the protocol that matches each description.
    - Translates IP address to MAC address: ARP
    - Reports network errors and provides ping: ICMP
    - Translates domain name to IP: DNS
    """
    matching = [
        ("Translates IP address to MAC address", "ARP",
         "Address Resolution Protocol - bridges L3 (IP) and L2 (MAC)"),
        ("Reports network errors and provides ping", "ICMP",
         "Internet Control Message Protocol - diagnostic and error reporting"),
        ("Translates domain name to IP", "DNS",
         "Domain Name System - hierarchical name resolution"),
    ]

    print("Protocol matching:")
    for description, protocol, explanation in matching:
        print(f"\n  Description: {description}")
        print(f"  Answer: {protocol}")
        print(f"  Detail: {explanation}")


def exercise_4():
    """
    Problem: Explain the sequence of the TCP 3-way handshake.

    Reasoning: The 3-way handshake establishes a reliable connection by
    synchronizing sequence numbers between client and server, ensuring
    both sides are ready to communicate.
    """
    import random

    # Simulate 3-way handshake with sequence numbers
    client_isn = random.randint(1000000, 9999999)
    server_isn = random.randint(1000000, 9999999)

    handshake = [
        {
            "step": 1,
            "direction": "Client -> Server",
            "flags": "SYN",
            "seq": client_isn,
            "ack": None,
            "description": "Client initiates connection with its Initial Sequence Number (ISN)",
        },
        {
            "step": 2,
            "direction": "Server -> Client",
            "flags": "SYN-ACK",
            "seq": server_isn,
            "ack": client_isn + 1,
            "description": "Server acknowledges client's SYN and sends its own ISN",
        },
        {
            "step": 3,
            "direction": "Client -> Server",
            "flags": "ACK",
            "seq": client_isn + 1,
            "ack": server_isn + 1,
            "description": "Client acknowledges server's SYN; connection established",
        },
    ]

    print("TCP 3-Way Handshake:")
    for step in handshake:
        ack_str = f"Ack={step['ack']}" if step["ack"] else "No Ack"
        print(f"\n  Step {step['step']}: {step['direction']}")
        print(f"    Flags: {step['flags']}, Seq={step['seq']}, {ack_str}")
        print(f"    Purpose: {step['description']}")

    print("\nAfter handshake: Both sides have synchronized sequence numbers.")
    print("Data transfer can begin in both directions.")


def exercise_5():
    """
    Problem: Describe the network communication process when accessing
    www.google.com in sequence.

    Reasoning: This traces a real-world web request through every protocol
    layer, showing how TCP/IP layers cooperate.
    """
    steps = [
        ("DNS Resolution", "Application",
         "Browser sends DNS query to resolve www.google.com -> IP address (e.g., 142.250.x.x)"),
        ("TCP Connection", "Transport",
         "3-way handshake (SYN, SYN-ACK, ACK) to establish TCP connection on port 443"),
        ("TLS Handshake", "Application/Transport",
         "ClientHello, ServerHello, certificate exchange, key agreement for HTTPS"),
        ("HTTP Request", "Application",
         "Send GET / HTTP/1.1 request with Host: www.google.com header"),
        ("IP Routing", "Internet",
         "IP packets routed through multiple hops (routers) to Google's servers"),
        ("Data Link Framing", "Network Access",
         "Each hop: Ethernet frame with MAC addresses (changes at each router)"),
        ("HTTP Response", "Application",
         "Server sends HTTP 200 OK with HTML content"),
        ("Page Rendering", "Application",
         "Browser parses HTML, requests additional resources (CSS, JS, images)"),
        ("TCP Teardown", "Transport",
         "4-way handshake (FIN, ACK, FIN, ACK) to close connection"),
    ]

    print("Communication process for accessing www.google.com:")
    for i, (step, layer, detail) in enumerate(steps, 1):
        print(f"\n  {i}. [{layer}] {step}")
        print(f"     {detail}")


def exercise_6():
    """
    Problem: Explain at least 5 differences between TCP and UDP.
    """
    differences = [
        ("Connection", "Connection-oriented (3-way handshake)", "Connectionless"),
        ("Reliability", "Guaranteed delivery (ACK, retransmission)", "Best-effort, no guarantees"),
        ("Ordering", "In-order delivery via sequence numbers", "No ordering guarantee"),
        ("Flow Control", "Sliding window mechanism", "None"),
        ("Congestion Control", "Slow start, congestion avoidance", "None"),
        ("Header Size", "20-60 bytes (options)", "8 bytes (fixed)"),
        ("Speed", "Slower due to overhead", "Faster, minimal overhead"),
        ("Broadcast", "Unicast only", "Supports broadcast/multicast"),
    ]

    print("TCP vs UDP - Key Differences:")
    print(f"  {'#':3s} {'Property':22s} {'TCP':35s} {'UDP':30s}")
    print("  " + "-" * 90)
    for i, (prop, tcp, udp) in enumerate(differences, 1):
        print(f"  {i:<3d} {prop:22s} {tcp:35s} {udp:30s}")


def exercise_7():
    """
    Problem: For each situation, which protocol is appropriate and why?
    (a) Online banking service
    (b) Real-time video conference
    (c) Large file download
    """
    scenarios = [
        {
            "scenario": "Online banking service",
            "protocol": "TCP",
            "reason": "Financial data must arrive complete and in order. "
                      "Loss of even one transaction record is unacceptable. "
                      "Reliability and data integrity are paramount.",
        },
        {
            "scenario": "Real-time video conference",
            "protocol": "UDP",
            "reason": "Low latency is critical for natural conversation. "
                      "A dropped frame is better than delayed playback. "
                      "Applications handle error recovery (e.g., skip, interpolate).",
        },
        {
            "scenario": "Large file download",
            "protocol": "TCP",
            "reason": "Every byte of the file must arrive correctly. "
                      "A corrupted file is useless. TCP's retransmission "
                      "ensures complete, error-free delivery.",
        },
    ]

    print("Protocol selection for each scenario:")
    for s in scenarios:
        print(f"\n  Scenario: {s['scenario']}")
        print(f"  Protocol: {s['protocol']}")
        print(f"  Reason: {s['reason']}")


def exercise_8():
    """
    Problem: Compare OSI 7-layer and TCP/IP 4-layer models and explain
    why TCP/IP became the Internet standard.
    """
    # Layer mapping between models
    mapping = [
        ("Application", [7, 6, 5], "Application + Presentation + Session"),
        ("Transport", [4], "Transport"),
        ("Internet", [3], "Network"),
        ("Network Access", [2, 1], "Data Link + Physical"),
    ]

    print("OSI to TCP/IP Layer Mapping:")
    print(f"  {'TCP/IP Layer':20s} {'OSI Layers':15s} {'OSI Names':40s}")
    print("  " + "-" * 75)
    for tcpip_name, osi_layers, osi_names in mapping:
        layers_str = ", ".join(f"L{l}" for l in osi_layers)
        print(f"  {tcpip_name:20s} {layers_str:15s} {osi_names:40s}")

    print("\nWhy TCP/IP became the Internet standard:")
    reasons = [
        "Practical: Built on working ARPANET code, not just theory",
        "Open: Free specification, no vendor lock-in",
        "Flexible: Works over any network technology",
        "First-mover: Deployed years before OSI protocols were ready",
        "Community: RFC process encouraged collaboration and rapid evolution",
    ]
    for r in reasons:
        print(f"  - {r}")


def exercise_9():
    """
    Problem: Analyze the scenario: traceroute shows '* * *' at the 5th hop.
    What are possible causes?
    """
    causes = [
        ("ICMP Blocked", "Router firewall drops ICMP Time Exceeded messages for security"),
        ("Rate Limiting", "Router rate-limits ICMP responses to prevent abuse"),
        ("High CPU Load", "Router deprioritizes ICMP processing under heavy load"),
        ("Network Congestion", "Packets or responses dropped due to queue overflow"),
        ("Security Policy", "Enterprise policy blocks traceroute probes"),
    ]

    print("Scenario: traceroute shows '* * *' at hop 5")
    print("=" * 60)
    print("\nPossible causes:")
    for i, (cause, detail) in enumerate(causes, 1):
        print(f"\n  {i}. {cause}")
        print(f"     {detail}")

    print("\nImportant: '* * *' does NOT necessarily mean the path is broken.")
    print("If subsequent hops respond, the router at hop 5 simply doesn't reply to probes.")


def exercise_10():
    """
    Problem: Explain why TCP's congestion control mechanisms
    (Slow Start, Congestion Avoidance) are necessary and how they work.

    Reasoning: Without congestion control, senders would flood the network,
    causing congestion collapse where throughput drops to near zero.
    """
    print("TCP Congestion Control Mechanisms:")
    print("=" * 60)

    print("\nWhy necessary:")
    print("  - Prevent network congestion collapse")
    print("  - Fair bandwidth sharing among competing flows")
    print("  - Adaptive to changing network conditions")

    # Simulate slow start and congestion avoidance
    ssthresh = 16  # Initial slow start threshold (MSS units)
    cwnd = 1       # Congestion window starts at 1 MSS
    mss = 1

    print(f"\nSimulation (ssthresh={ssthresh}, initial cwnd={cwnd}):")
    print(f"  {'RTT':5s} {'cwnd':8s} {'Phase':25s}")
    print("  " + "-" * 40)

    for rtt in range(1, 21):
        if cwnd < ssthresh:
            phase = "Slow Start (exponential)"
            cwnd = cwnd * 2  # Double every RTT
        else:
            phase = "Congestion Avoidance (linear)"
            cwnd = cwnd + mss  # Add 1 MSS per RTT

        print(f"  {rtt:<5d} {cwnd:<8d} {phase}")

        # Simulate packet loss at cwnd=32
        if cwnd >= 32:
            print(f"\n  ** Timeout at cwnd={cwnd}! **")
            ssthresh = cwnd // 2
            cwnd = 1
            print(f"  New ssthresh={ssthresh}, cwnd reset to {cwnd}")
            break

    print("\nKey algorithms:")
    print("  Slow Start: cwnd doubles every RTT (exponential growth)")
    print("  Congestion Avoidance: cwnd increases by 1 MSS per RTT (linear)")
    print("  On timeout: ssthresh = cwnd/2, cwnd = 1 MSS")
    print("  On 3 dup ACKs: ssthresh = cwnd/2, cwnd = ssthresh + 3 (Fast Recovery)")


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
