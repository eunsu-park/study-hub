"""
Exercises for Lesson 02: OSI 7-Layer Model
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem: List the OSI 7 layers in order from bottom to top.

    Reasoning: The OSI model provides a conceptual framework for understanding
    how different network protocols interact at each abstraction level.
    """
    layers = [
        (1, "Physical", "Bit transmission over physical media"),
        (2, "Data Link", "Reliable hop-to-hop delivery, MAC addressing"),
        (3, "Network", "Logical addressing, routing between networks"),
        (4, "Transport", "End-to-end delivery, segmentation, flow control"),
        (5, "Session", "Dialog control, synchronization"),
        (6, "Presentation", "Data format translation, encryption, compression"),
        (7, "Application", "User interface, application protocols"),
    ]

    print("OSI 7 Layers (bottom to top):")
    for num, name, function in layers:
        print(f"  Layer {num}: {name:15s} - {function}")

    # Mnemonic
    print("\nMnemonic: 'Please Do Not Throw Sausage Pizza Away'")


def exercise_2():
    """
    Problem: Match protocols to their corresponding layers.
    (a) HTTP -> Application Layer
    (b) TCP -> Transport Layer
    (c) IP -> Network Layer
    (d) Ethernet -> Data Link Layer
    """
    protocol_layer_map = {
        "HTTP":     {"layer": 7, "layer_name": "Application"},
        "TCP":      {"layer": 4, "layer_name": "Transport"},
        "IP":       {"layer": 3, "layer_name": "Network"},
        "Ethernet": {"layer": 2, "layer_name": "Data Link"},
    }

    print("Protocol to Layer matching:")
    for protocol, info in protocol_layer_map.items():
        print(f"  {protocol:10s} -> Layer {info['layer']} ({info['layer_name']})")


def exercise_3():
    """
    Problem: Choose the correct PDU matching.
    Transport Layer: Segment
    Network Layer: Packet
    Data Link Layer: Frame
    """
    pdu_map = {
        "Application": "Data",
        "Presentation": "Data",
        "Session": "Data",
        "Transport": "Segment",
        "Network": "Packet",
        "Data Link": "Frame",
        "Physical": "Bit",
    }

    print("PDU (Protocol Data Unit) at each layer:")
    for layer, pdu in pdu_map.items():
        print(f"  {layer:15s} -> {pdu}")

    print("\nAnswer:")
    print("  Transport Layer: Segment")
    print("  Network Layer: Packet")
    print("  Data Link Layer: Frame")


def exercise_4():
    """
    Problem: Explain the order in which headers are added during encapsulation.

    Reasoning: Encapsulation is how data moves down the stack -- each layer wraps
    the upper layer's PDU with its own header (and sometimes trailer), creating
    a layered structure that can be independently processed at each hop.
    """
    print("Encapsulation process (top to bottom):")

    data = "Hello, World!"
    steps = [
        ("Application", f"Data: '{data}'", ""),
        ("Transport", f"[TCP Header | '{data}']", "Adds: src/dst port, seq number"),
        ("Network", f"[IP Header | TCP Header | '{data}']", "Adds: src/dst IP, TTL"),
        ("Data Link", f"[Eth Header | IP Header | TCP Header | '{data}' | FCS]", "Adds: src/dst MAC, FCS trailer"),
        ("Physical", "01101000 01100101 01101100 ...", "Converts to bits"),
    ]

    for layer, pdu, note in steps:
        print(f"\n  {layer} Layer:")
        print(f"    PDU: {pdu}")
        if note:
            print(f"    Note: {note}")


def exercise_5():
    """
    Problem: Estimate the layer where the problem occurred.
    (a) Connected cable but LED doesn't light up
    (b) Can ping other PCs on same network but no internet
    (c) Can access web page but cannot log in

    Reasoning: Troubleshooting networks follows a bottom-up approach through
    the OSI layers -- physical issues first, then link, network, and up.
    """
    scenarios = [
        {
            "symptom": "Connected cable but LED doesn't light up",
            "layer": 1,
            "layer_name": "Physical",
            "diagnosis": "Cable fault, NIC failure, or port disabled",
            "checks": ["Verify cable integrity", "Try different port", "Check NIC LED"],
        },
        {
            "symptom": "Can ping same-network PCs but no internet",
            "layer": 3,
            "layer_name": "Network",
            "diagnosis": "Default gateway misconfigured or routing issue",
            "checks": ["Check default gateway", "Verify routing table", "Test DNS resolution"],
        },
        {
            "symptom": "Can access web page but cannot log in",
            "layer": 7,
            "layer_name": "Application (or Session)",
            "diagnosis": "Authentication failure, session management issue, or application bug",
            "checks": ["Check credentials", "Clear cookies/cache", "Verify HTTPS/TLS"],
        },
    ]

    print("Layer-based troubleshooting:")
    for s in scenarios:
        print(f"\n  Symptom: {s['symptom']}")
        print(f"  Likely Layer: {s['layer']} ({s['layer_name']})")
        print(f"  Diagnosis: {s['diagnosis']}")
        print(f"  Checks: {', '.join(s['checks'])}")


def exercise_6():
    """
    Problem: Explain the role of each OSI layer in HTTP and HTTPS communication.

    Reasoning: Tracing a real protocol through all 7 layers demonstrates how
    each layer adds specific functionality to enable end-to-end communication.
    """
    http_layers = [
        (7, "Application", "Generate/process HTTP requests and responses"),
        (6, "Presentation", "TLS/SSL encryption (HTTPS), data compression"),
        (5, "Session", "TCP connection management, keep-alive sessions"),
        (4, "Transport", "TCP segmentation, port numbers (80/443), flow control"),
        (3, "Network", "IP packet routing between source and destination"),
        (2, "Data Link", "Ethernet framing, MAC addressing per hop"),
        (1, "Physical", "Bit transmission over wire/fiber/wireless"),
    ]

    print("OSI layer roles in HTTP/HTTPS communication:")
    for num, name, role in http_layers:
        https_note = " *" if num == 6 else ""
        print(f"  L{num} {name:14s}: {role}{https_note}")

    print("\n  * HTTPS adds TLS encryption at the Presentation layer.")
    print("    HTTP (plaintext) skips encryption at L6.")


def exercise_7():
    """
    Problem: Explain TCP vs UDP differences from OSI model perspective.

    Reasoning: Both TCP and UDP operate at Layer 4 (Transport) but provide
    fundamentally different service models -- reliable streams vs unreliable datagrams.
    """
    comparison = {
        "Property": [
            "Connection", "Reliability", "Ordering", "Flow Control",
            "Overhead", "Speed", "PDU Name", "Use Cases",
        ],
        "TCP": [
            "Connection-oriented (3-way handshake)", "Guaranteed delivery (ACK/retransmit)",
            "In-order delivery guaranteed", "Sliding window + congestion control",
            "20+ bytes header", "Slower (overhead)", "Segment",
            "Web (HTTP), Email (SMTP), File transfer (FTP)",
        ],
        "UDP": [
            "Connectionless", "Best-effort (no guarantees)",
            "No ordering guarantee", "None",
            "8 bytes header", "Faster (minimal overhead)", "Datagram",
            "Streaming, Gaming, DNS, VoIP",
        ],
    }

    print("TCP vs UDP at the Transport Layer (L4):")
    print("-" * 70)
    for i, prop in enumerate(comparison["Property"]):
        print(f"\n  {prop}:")
        print(f"    TCP: {comparison['TCP'][i]}")
        print(f"    UDP: {comparison['UDP'][i]}")


def exercise_8():
    """
    Problem: Indicate which OSI layers each device processes.
    [PC] ---[Hub]---[Switch]---[Router]---[Firewall]---[Server]

    Reasoning: Each network device operates up to a specific layer, determining
    what information it can inspect and use for forwarding decisions.
    """
    devices = {
        "Hub":      {"layers": [1], "description": "Repeats electrical signals to all ports"},
        "Switch":   {"layers": [1, 2], "description": "Forwards frames based on MAC address table"},
        "Router":   {"layers": [1, 2, 3], "description": "Routes packets based on IP routing table"},
        "Firewall": {"layers": [1, 2, 3, 4, 5, 6, 7], "description": "Inspects up to L7 depending on type"},
        "PC/Server": {"layers": [1, 2, 3, 4, 5, 6, 7], "description": "Full stack processing"},
    }

    print("Device layer processing:")
    for device, info in devices.items():
        layers_str = ", ".join(f"L{l}" for l in info["layers"])
        print(f"\n  {device:12s}: Processes {layers_str}")
        print(f"               {info['description']}")

    print("\nNote: L7 firewalls (WAF/NGFW) inspect application data.")
    print("      L4 firewalls only inspect transport headers (ports, flags).")


def exercise_9():
    """
    Problem: Explain why encapsulation and decapsulation are necessary.

    Reasoning: The layered approach enables modularity -- each layer can evolve
    independently, different technologies can coexist, and troubleshooting
    is simplified by isolating issues to specific layers.
    """
    benefits = [
        ("Layer Independence", "Changes in one layer don't require changes in other layers"),
        ("Standardized Interfaces", "Layers communicate through well-defined interfaces"),
        ("Interoperability", "Products from different vendors can work together"),
        ("Modular Design", "Each layer can be developed and updated independently"),
        ("Troubleshooting", "Problems can be isolated to specific layers"),
    ]

    print("Why encapsulation/decapsulation is necessary:")
    for i, (benefit, explanation) in enumerate(benefits, 1):
        print(f"\n  {i}. {benefit}")
        print(f"     {explanation}")

    # Demonstrate with a simple example
    print("\n\nExample: Changing from Ethernet (L2) to Wi-Fi (L2)")
    print("  - Only Layer 2 framing changes")
    print("  - Layer 3+ (IP, TCP, HTTP) remains completely unchanged")
    print("  - This is only possible because of encapsulation boundaries")


def exercise_10():
    """
    Problem: Explain the differences between the OSI model and TCP/IP model.

    Reasoning: OSI is a theoretical reference model while TCP/IP is a practical
    implementation model. Understanding both helps map protocols to functions.
    """
    comparison = [
        ("Layers", "7 layers", "4 layers"),
        ("Origin", "ISO standard (theoretical)", "DARPA/Internet (practical)"),
        ("Development", "Model first, then protocols", "Protocols first, then model"),
        ("Session/Presentation", "Separate dedicated layers", "Combined into Application"),
        ("Network Access", "Physical + Data Link separate", "Combined into Network Access"),
        ("Adoption", "Reference/teaching model", "Actual Internet standard"),
        ("Protocol Binding", "Protocol-independent", "Tightly coupled to TCP/IP suite"),
    ]

    print("OSI Model vs TCP/IP Model:")
    print("-" * 70)
    print(f"  {'Aspect':25s} {'OSI':25s} {'TCP/IP':25s}")
    print("-" * 70)
    for aspect, osi, tcpip in comparison:
        print(f"  {aspect:25s} {osi:25s} {tcpip:25s}")

    print("\nWhy TCP/IP became the Internet standard:")
    print("  - Practical approach: built on working ARPANET code")
    print("  - Open and free: no licensing, anyone could implement")
    print("  - Flexible: supports diverse network technologies")
    print("  - First-mover advantage: deployed before OSI was finalized")


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
