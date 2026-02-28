"""
Exercises for Lesson 17: Practical Network Tools
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: ping
    - What 3 pieces of information can you get from ping?
    - If TTL is 116 and original TTL was 128, how many routers traversed?
    """
    print("Information from ping:")
    info = [
        ("Reachability", "Whether the destination host is alive and responding"),
        ("Round-Trip Time (RTT)", "Latency in milliseconds (min/avg/max/stddev)"),
        ("Packet Loss", "Percentage of packets that didn't get a response"),
    ]
    for i, (name, detail) in enumerate(info, 1):
        print(f"  {i}. {name}: {detail}")

    print("\n  Additional info from TTL:")
    original_ttl = 128
    received_ttl = 116
    hops = original_ttl - received_ttl
    print(f"    Received TTL: {received_ttl}")
    print(f"    Original TTL: {original_ttl} (common for Windows)")
    print(f"    Routers traversed: {original_ttl} - {received_ttl} = {hops} hops")

    print("\n  Common default TTL values:")
    ttl_defaults = {"Linux": 64, "Windows": 128, "macOS": 64, "Cisco": 255}
    for os_name, ttl in ttl_defaults.items():
        print(f"    {os_name:10s}: {ttl}")


def exercise_2():
    """
    Problem 2: traceroute
    - How traceroute uses TTL.
    - What does '* * *' output mean?
    """
    print("How traceroute uses TTL:")
    print()
    print("  Mechanism:")
    print("    1. Send packet with TTL=1 -> first router decrements to 0")
    print("       Router drops packet, sends ICMP Time Exceeded back")
    print("       Now we know hop 1's IP address and RTT")
    print("    2. Send packet with TTL=2 -> reaches second router")
    print("       Same process reveals hop 2")
    print("    3. Repeat, incrementing TTL each time")
    print("    4. When destination reached, get ICMP Port Unreachable (UDP)")
    print("       or Echo Reply (ICMP mode)")

    # Simulate traceroute output
    print("\n  Simulated output:")
    hops = [
        (1, "192.168.1.1", [1.2, 0.9, 1.1]),
        (2, "10.0.0.1", [5.3, 4.8, 5.1]),
        (3, None, None),  # * * *
        (4, "172.16.0.1", [15.2, 14.8, 15.5]),
        (5, "8.8.8.8", [20.1, 19.5, 20.3]),
    ]

    for hop, ip, rtts in hops:
        if ip:
            rtt_str = "  ".join(f"{r:.1f} ms" for r in rtts)
            print(f"    {hop:2d}  {ip:15s}  {rtt_str}")
        else:
            print(f"    {hop:2d}  * * *")

    print("\n  '* * *' means:")
    print("    - Router at that hop did NOT respond")
    print("    - Possible reasons:")
    print("      1. Router blocks ICMP Time Exceeded messages (firewall)")
    print("      2. Router rate-limits ICMP responses")
    print("      3. Router has high CPU load, drops ICMP processing")
    print("      4. Network congestion caused packet loss")
    print("    - Does NOT necessarily mean the path is broken")


def exercise_3():
    """
    Problem 3: netstat/ss
    - LISTEN vs ESTABLISHED states.
    - Command to find processes using port 80.
    """
    print("TCP Connection States:")

    print("\n  LISTEN:")
    print("    - Server socket waiting for incoming connections")
    print("    - Port is open and ready to accept clients")
    print("    - Example: Web server listening on port 80")

    print("\n  ESTABLISHED:")
    print("    - Active connection with data flowing")
    print("    - 3-way handshake completed successfully")
    print("    - Example: Browser connected to web server")

    # Show other common states
    states = {
        "LISTEN": "Waiting for connections",
        "SYN_SENT": "Client sent SYN, waiting for SYN-ACK",
        "SYN_RECV": "Server received SYN, sent SYN-ACK",
        "ESTABLISHED": "Connection active",
        "FIN_WAIT_1": "Sent FIN, waiting for ACK",
        "FIN_WAIT_2": "Received ACK for FIN, waiting for peer's FIN",
        "TIME_WAIT": "Waiting 2*MSL after closing (prevents stale packets)",
        "CLOSE_WAIT": "Received FIN, waiting for application to close",
        "LAST_ACK": "Sent FIN, waiting for final ACK",
        "CLOSED": "Connection terminated",
    }

    print(f"\n  All TCP states:")
    for state, description in states.items():
        print(f"    {state:15s}: {description}")

    print(f"\n  Commands to find processes on port 80:")
    print(f"    Linux:   ss -tulnp | grep :80")
    print(f"    Linux:   lsof -i :80")
    print(f"    macOS:   lsof -iTCP -sTCP:LISTEN -P | grep :80")
    print(f"    Windows: netstat -ano | findstr :80")


def exercise_4():
    """
    Problem 4: DNS Tools
    - Differences between nslookup and dig.
    - MX vs A record purpose.
    """
    print("nslookup vs dig:")
    comparison = [
        ("Output", "Simple, user-friendly", "Detailed, parseable format"),
        ("Scripting", "Harder to parse", "Easy to parse, grep-friendly"),
        ("Features", "Basic lookups", "+trace, +short, @server, many options"),
        ("Default", "Uses system resolver", "Can specify any DNS server"),
        ("Status", "Deprecated on some systems", "Preferred modern tool"),
    ]

    print(f"  {'Aspect':12s} {'nslookup':30s} {'dig':30s}")
    print(f"  {'-'*72}")
    for aspect, ns, d in comparison:
        print(f"  {aspect:12s} {ns:30s} {d:30s}")

    print("\n\nMX vs A Record Purpose:")
    print("  A Record:")
    print("    Purpose: Map hostname to IPv4 address")
    print("    Example: www.example.com -> 93.184.216.34")
    print("    Used for: Web servers, any host resolution")

    print("\n  MX Record:")
    print("    Purpose: Specify mail server for a domain")
    print("    Example: example.com -> 10 mail.example.com")
    print("    Used for: Email delivery routing")
    print("    Has priority value (lower = tried first)")


def exercise_5():
    """
    Problem 5: tcpdump
    Explain: sudo tcpdump -i eth0 'tcp port 80 and host 192.168.1.100'
    """
    print("tcpdump Filter Analysis:")
    print("  Command: sudo tcpdump -i eth0 'tcp port 80 and host 192.168.1.100'")
    print()

    parts = [
        ("sudo", "Run with root privileges (required for packet capture)"),
        ("tcpdump", "Command-line packet analyzer"),
        ("-i eth0", "Capture on interface eth0"),
        ("tcp", "Only TCP protocol packets"),
        ("port 80", "Source OR destination port 80 (HTTP)"),
        ("and", "Both conditions must be true"),
        ("host 192.168.1.100", "Source OR destination IP is 192.168.1.100"),
    ]

    print("  Breakdown:")
    for part, meaning in parts:
        print(f"    {part:25s} -> {meaning}")

    print(f"\n  Effect: Captures only HTTP (TCP port 80) traffic")
    print(f"  to or from the host 192.168.1.100 on the eth0 interface.")
    print(f"\n  Useful for: Debugging HTTP requests from a specific client")
    print(f"  or monitoring a web server's traffic with one client.")


def exercise_6():
    """
    Problem 6: Troubleshooting Scenarios
    a) Web server not responding
    b) Domain works but specific site doesn't
    c) Intermittent packet loss
    """
    scenarios = [
        {
            "scenario": "Web server (192.168.1.100) not responding",
            "steps": [
                ("ping 192.168.1.100", "Check basic connectivity (L3)"),
                ("nc -zv 192.168.1.100 80", "Check if port 80 is open (L4)"),
                ("curl -v http://192.168.1.100", "Check HTTP response (L7)"),
                ("ss -tuln | grep :80", "On server: verify web service is listening"),
                ("sudo tcpdump -i eth0 host 192.168.1.100", "Capture traffic to diagnose"),
            ],
        },
        {
            "scenario": "Domain access works but specific site doesn't",
            "steps": [
                ("dig example.com", "Verify DNS resolution (correct IP?)"),
                ("ping example.com", "Check network path to the IP"),
                ("curl -I https://example.com", "Check HTTP headers and status code"),
                ("openssl s_client -connect example.com:443", "Check TLS certificate"),
                ("traceroute example.com", "Check routing path for anomalies"),
            ],
        },
        {
            "scenario": "Intermittent packet loss",
            "steps": [
                ("mtr -r google.com", "Combined ping+traceroute, shows per-hop loss"),
                ("ping -c 100 gateway_ip", "Extended ping to local gateway"),
                ("ethtool eth0 | grep -i error", "Check interface error counters"),
                ("dmesg | grep -i eth", "Check kernel logs for NIC errors"),
                ("iperf3 -c server_ip", "Bandwidth test to isolate capacity issues"),
            ],
        },
    ]

    print("Troubleshooting Methodology:")
    for s in scenarios:
        print(f"\n  Scenario: {s['scenario']}")
        print(f"  Diagnostic steps:")
        for cmd, purpose in s["steps"]:
            print(f"    $ {cmd:45s} # {purpose}")


def exercise_7():
    """
    Problem 7: Wireshark
    - Filter for TCP 3-way handshake.
    - Filter for HTTP 500+ status codes.
    """
    print("Wireshark Display Filters:")

    print("\n  TCP 3-way handshake filter:")
    print("    SYN only:    tcp.flags.syn == 1 && tcp.flags.ack == 0")
    print("    SYN-ACK:     tcp.flags.syn == 1 && tcp.flags.ack == 1")
    print("    All three:   tcp.flags.syn == 1 || (tcp.flags.ack == 1 && tcp.seq == 1)")

    print("\n  HTTP 500+ status codes:")
    print("    Exact 500:   http.response.code == 500")
    print("    500 and up:  http.response.code >= 500")
    print("    All errors:  http.response.code >= 400")

    print("\n  Other useful filters:")
    useful_filters = [
        ("ip.addr == 192.168.1.1", "Traffic to/from specific IP"),
        ("tcp.port == 443", "HTTPS traffic"),
        ("dns", "All DNS queries and responses"),
        ("http.request.method == POST", "HTTP POST requests only"),
        ("tcp.analysis.retransmission", "TCP retransmissions (loss indicator)"),
        ("frame.time_delta > 1", "Frames with >1 second gap (slowness)"),
    ]
    for filter_str, purpose in useful_filters:
        print(f"    {filter_str:45s} # {purpose}")


def exercise_8():
    """
    Problem 8: Comprehensive Troubleshooting
    Step-by-step approach when web service is slow.
    """
    print("Systematic Web Service Slowness Diagnosis:")
    print("  (Bottom-up approach through network layers)")

    steps = [
        ("1. Physical/Network (L1-L3)", [
            "ping web_server - check basic connectivity and RTT",
            "traceroute web_server - identify slow hops",
            "mtr web_server - continuous monitoring of each hop",
            "Check for interface errors: ethtool, ip -s link",
        ]),
        ("2. Transport (L4)", [
            "ss -tan | grep web_server - check connection states",
            "Look for many TIME_WAIT or SYN_SENT (connection issues)",
            "tcpdump: check for retransmissions, window sizing",
            "Check TCP metrics: ss -i (cwnd, rtt, retrans count)",
        ]),
        ("3. Application (L7)", [
            "curl -w '@timing' - measure DNS, connect, TTFB, total",
            "Check server logs: access.log, error.log",
            "Monitor server resources: CPU, memory, disk I/O",
            "Check application metrics: request queue, DB query time",
        ]),
        ("4. DNS", [
            "dig +stats web_domain - check query time",
            "dig @8.8.8.8 vs dig @local_dns - compare resolvers",
            "Check for DNS caching issues (low TTL, cache miss)",
        ]),
        ("5. TLS/SSL", [
            "openssl s_client -connect server:443 - check handshake time",
            "Check certificate chain length (long chain = slow)",
            "Verify OCSP stapling is enabled",
        ]),
    ]

    for phase, checks in steps:
        print(f"\n  {phase}:")
        for check in checks:
            print(f"    - {check}")

    print("\n  Pro tip: Use 'curl -w' for quick timing breakdown:")
    print("    curl -o /dev/null -s -w 'DNS: %{time_namelookup}s\\n"
          "Connect: %{time_connect}s\\nTTFB: %{time_starttransfer}s\\n"
          "Total: %{time_total}s\\n' https://example.com")


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
