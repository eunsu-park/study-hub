"""
Exercises for Lesson 01: IoT Overview
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates IoT system design decisions using Python data structures.
"""


# === Problem 1: IoT System Design ===
# Problem: Design a smart parking lot system. Include:
# - Types of sensors needed
# - Communication protocol selection (with reasoning)
# - Edge vs cloud processing distribution

def problem_1():
    """Solution: Design a smart parking lot IoT system."""

    # A smart parking lot requires detecting vehicle presence in each space,
    # guiding drivers to open spots, and reporting occupancy to a central system.
    # The design must balance real-time responsiveness (for guidance displays)
    # with cost efficiency (hundreds of sensors per lot).

    system_design = {
        "sensors": {
            "Ultrasonic sensor (HC-SR04)": {
                "purpose": "Detect vehicle presence in each parking space",
                "placement": "Mounted overhead or ground-embedded per space",
                "reason": "Reliable distance measurement distinguishes empty vs occupied; "
                          "works in all lighting conditions unlike cameras",
            },
            "Infrared (IR) break-beam sensor": {
                "purpose": "Count vehicles entering and exiting at gate barriers",
                "placement": "Entry and exit lanes",
                "reason": "Fast trigger for gate control; pairs give direction detection",
            },
            "Camera (optional)": {
                "purpose": "License plate recognition for automated access",
                "placement": "Entry/exit gates",
                "reason": "Enables ticketless parking and enforcement",
            },
            "Environmental sensor (DHT22)": {
                "purpose": "Monitor temperature and humidity in underground lots",
                "placement": "One per zone for ventilation control",
                "reason": "Safety: detect CO buildup via correlated temp/humidity anomalies",
            },
        },
        "communication_protocol": {
            "primary": "MQTT",
            "reasoning": [
                "Hundreds of sensors publishing small payloads (occupied/empty) at low frequency",
                "Pub/sub model lets the edge gateway, mobile app, and dashboard all subscribe "
                "without coupling to individual sensors",
                "QoS 1 guarantees at-least-once delivery for occupancy changes",
                "Low bandwidth: each message is a few bytes, far less than HTTP overhead",
                "Retained messages let new subscribers immediately know each spot's status",
            ],
            "secondary": "HTTP/REST",
            "secondary_reasoning": [
                "Mobile app queries the REST API for lot overview and navigation",
                "Admin dashboard fetches historical statistics via HTTP GET",
                "HTTP is standard for user-facing interfaces; MQTT handles device-to-server",
            ],
        },
        "edge_vs_cloud": {
            "edge_processing": [
                "Occupancy detection: each sensor node decides occupied/empty locally "
                "(threshold on distance reading) to avoid streaming raw distance data",
                "Guidance display updates: edge gateway computes available spots per zone "
                "and drives LED signs without waiting for cloud round-trip",
                "Gate control: IR sensor triggers gate open/close immediately at the edge "
                "(latency-critical; cannot depend on cloud connectivity)",
                "Data aggregation: edge gateway batches individual sensor readings and "
                "publishes zone-level summaries to reduce bandwidth by ~95%",
            ],
            "cloud_processing": [
                "Historical analytics: occupancy trends by hour/day/month for pricing models",
                "License plate recognition model inference (GPU resources in cloud)",
                "Cross-lot management: compare occupancy across multiple parking structures",
                "Billing and payment processing (requires integration with payment APIs)",
                "Machine learning: predict peak times, optimize dynamic pricing",
            ],
        },
    }

    print("=== Smart Parking Lot IoT System Design ===\n")

    print("--- Sensors ---")
    for sensor, details in system_design["sensors"].items():
        print(f"\n  {sensor}")
        print(f"    Purpose: {details['purpose']}")
        print(f"    Placement: {details['placement']}")
        print(f"    Reason: {details['reason']}")

    print("\n--- Communication Protocol ---")
    proto = system_design["communication_protocol"]
    print(f"\n  Primary: {proto['primary']}")
    for r in proto["reasoning"]:
        print(f"    - {r}")
    print(f"\n  Secondary: {proto['secondary']}")
    for r in proto["secondary_reasoning"]:
        print(f"    - {r}")

    print("\n--- Edge vs Cloud Processing ---")
    evc = system_design["edge_vs_cloud"]
    print("\n  Edge Processing:")
    for item in evc["edge_processing"]:
        print(f"    - {item}")
    print("\n  Cloud Processing:")
    for item in evc["cloud_processing"]:
        print(f"    - {item}")


# === Problem 2: Protocol Selection ===
# Problem: Select appropriate protocols for the following scenarios and explain why:
# 1. Battery-powered remote temperature sensor
# 2. Real-time security camera video streaming
# 3. Smart lighting control system

def problem_2():
    """Solution: Protocol selection for three IoT scenarios."""

    scenarios = [
        {
            "scenario": "Battery-powered remote temperature sensor",
            "protocol": "CoAP (Constrained Application Protocol) over UDP",
            "alternatives_considered": ["MQTT", "HTTP"],
            "reasoning": [
                "CoAP is designed for constrained devices with limited CPU, memory, and power",
                "UDP-based: no TCP handshake overhead saves power on each transmission",
                "Supports GET/PUT semantics like HTTP but with minimal packet size (~4 bytes header)",
                "Observe option allows the sensor to notify subscribers of changes "
                "without continuous polling (similar to MQTT notify but lighter)",
                "If a broker is already deployed, MQTT-SN (Sensor Networks variant) is also viable: "
                "it runs over UDP and removes the TCP requirement that standard MQTT imposes",
            ],
            "why_not_http": "HTTP over TCP is too heavy for a battery device sending small payloads; "
                           "the TCP handshake alone consumes more energy than the data transfer",
        },
        {
            "scenario": "Real-time security camera video streaming",
            "protocol": "RTSP/RTP over UDP (or WebSocket for web-based viewing)",
            "alternatives_considered": ["MQTT", "HTTP"],
            "reasoning": [
                "Video streams require high throughput (1-10 Mbps); MQTT's small-message design "
                "is not suited for continuous binary streams",
                "RTSP/RTP is purpose-built for media streaming with low latency",
                "UDP transport tolerates occasional packet loss (a few dropped video frames "
                "are preferable to the latency spikes TCP retransmission would cause)",
                "For browser-based access, WebSocket over TCP provides persistent "
                "bidirectional streaming compatible with web clients",
                "HTTP is used only for camera configuration (e.g., REST API to change resolution), "
                "not for the video stream itself",
            ],
            "why_not_mqtt": "MQTT is designed for small telemetry messages (kilobytes), not "
                           "megabyte-per-second video data; it would overwhelm the broker",
        },
        {
            "scenario": "Smart lighting control system",
            "protocol": "MQTT (with QoS 1)",
            "alternatives_considered": ["HTTP", "CoAP", "BLE Mesh"],
            "reasoning": [
                "Pub/sub model maps naturally to lighting: a controller publishes to "
                "'room/living-room/light/command', and all lights in that room subscribe",
                "Retained messages ensure a new subscriber immediately knows the current state",
                "QoS 1 guarantees command delivery so lights do not miss on/off instructions",
                "Lightweight: smart light microcontrollers (ESP32) handle MQTT easily",
                "Wildcard subscriptions enable group control: 'room/+/light/command' targets "
                "all rooms without listing each one",
                "For local mesh without WiFi infrastructure, BLE Mesh is a strong alternative "
                "(direct phone control, no router needed); choose based on infrastructure",
            ],
            "why_not_http": "HTTP requires the light to run an HTTP server or the controller to "
                           "know each light's IP; MQTT decouples sender from receiver via the broker",
        },
    ]

    print("=== Protocol Selection for IoT Scenarios ===\n")

    for i, s in enumerate(scenarios, 1):
        print(f"--- Scenario {i}: {s['scenario']} ---")
        print(f"  Recommended: {s['protocol']}")
        print(f"  Alternatives considered: {', '.join(s['alternatives_considered'])}")
        print("  Reasoning:")
        for r in s["reasoning"]:
            print(f"    - {r}")
        for key in ["why_not_http", "why_not_mqtt"]:
            if key in s:
                label = key.replace("why_not_", "Why not ").replace("_", " ").capitalize()
                print(f"  {label}: {s[key]}")
        print()


# === Problem 3: Security Analysis ===
# Problem: List 3 potential security vulnerabilities for a home smart door lock
# and propose countermeasures for each.

def problem_3():
    """Solution: Security analysis for a smart door lock."""

    # A smart door lock is an extremely security-sensitive IoT device because
    # compromise directly grants physical access to a home. The threat model
    # spans all three IoT layers: device, network, and cloud/application.

    vulnerabilities = [
        {
            "vulnerability": "Default or weak credentials on the lock's management interface",
            "attack_vector": "Attacker discovers the lock exposes a web UI or BLE management "
                            "service using factory-default credentials (e.g., admin/admin). "
                            "They connect and unlock the door remotely.",
            "real_world_example": "Many consumer IoT devices ship with well-known default passwords "
                                 "listed in product manuals available online",
            "countermeasures": [
                "Force a unique password/PIN setup during first-time configuration (block "
                "operation until defaults are changed)",
                "Use certificate-based authentication instead of passwords for BLE pairing",
                "Implement account lockout after 5 failed attempts with exponential backoff",
                "Disable any management interface that is not actively needed",
            ],
        },
        {
            "vulnerability": "Replay attack on the BLE or WiFi unlock command",
            "attack_vector": "Attacker eavesdrops on the wireless communication between the "
                            "smartphone app and the lock, captures the unlock command packet, "
                            "and retransmits it later to unlock the door without authorization.",
            "real_world_example": "BLE sniffers (e.g., Ubertooth) can capture unencrypted BLE packets; "
                                 "WiFi packet capture tools like Wireshark can record HTTP requests",
            "countermeasures": [
                "Use challenge-response authentication: lock sends a random nonce, phone signs "
                "it with a shared secret; each nonce is single-use so replays are rejected",
                "Enable BLE Secure Connections (LE Secure Connections with ECDH key exchange) "
                "for encrypted, replay-resistant communication",
                "For WiFi: use TLS for all command traffic; TLS session keys change per connection "
                "making captured packets useless for replay",
                "Add timestamp or sequence number to each command; reject commands outside "
                "a narrow time window (e.g., 30 seconds)",
            ],
        },
        {
            "vulnerability": "Firmware tampering or extraction via physical access",
            "attack_vector": "Attacker removes the lock from the door, connects to the debug "
                            "port (JTAG/UART), dumps the firmware, extracts encryption keys "
                            "or credentials, then installs modified firmware that always unlocks.",
            "real_world_example": "Researchers have demonstrated firmware extraction from smart locks "
                                 "using inexpensive tools like a $10 UART adapter",
            "countermeasures": [
                "Enable Secure Boot: only firmware signed by the manufacturer's private key "
                "can execute, preventing installation of tampered images",
                "Disable JTAG/UART debug interfaces in production firmware (blow eFuses "
                "or use hardware security modules to lock down debug access)",
                "Store encryption keys in a hardware secure element (e.g., ATECC608) "
                "rather than in flash memory where they can be read out",
                "Implement tamper detection: an accelerometer or contact switch detects "
                "physical removal and triggers an alert to the homeowner's phone",
            ],
        },
    ]

    print("=== Security Analysis: Smart Door Lock ===\n")

    for i, v in enumerate(vulnerabilities, 1):
        print(f"--- Vulnerability {i}: {v['vulnerability']} ---")
        print(f"  Attack vector: {v['attack_vector']}")
        print(f"  Real-world example: {v['real_world_example']}")
        print("  Countermeasures:")
        for c in v["countermeasures"]:
            print(f"    - {c}")
        print()


# === Run All Problems ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 01: IoT Overview - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Problem 1: IoT System Design")
    print("-" * 50)
    problem_1()

    print("\n\n>>> Problem 2: Protocol Selection")
    print("-" * 50)
    problem_2()

    print("\n\n>>> Problem 3: Security Analysis")
    print("-" * 50)
    problem_3()
