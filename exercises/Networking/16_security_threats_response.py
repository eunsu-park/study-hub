"""
Exercises for Lesson 16: Security Threats and Response
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: Sniffing/Spoofing
    - Difference between sniffing and spoofing.
    - Operating principle of ARP spoofing.
    """
    print("Sniffing vs Spoofing:")

    print("\n  Sniffing (Passive Attack):")
    print("    - Capturing and reading network traffic")
    print("    - Attacker is a silent observer")
    print("    - Tools: Wireshark, tcpdump")
    print("    - Defense: Encryption (TLS, VPN)")

    print("\n  Spoofing (Active Attack):")
    print("    - Forging/impersonating identity (IP, MAC, DNS)")
    print("    - Attacker actively modifies or injects packets")
    print("    - Tools: arpspoof, hping3")
    print("    - Defense: Authentication, digital signatures")

    print("\n\nARP Spoofing Principle:")
    steps = [
        "1. Network has: Gateway (192.168.1.1, MAC=GW:GW:GW) and Victim (192.168.1.10)",
        "2. Attacker sends forged ARP Reply to Victim:",
        "   '192.168.1.1 is at ATTACKER:MAC:ADDR' (lies about gateway's MAC)",
        "3. Victim updates ARP cache: 192.168.1.1 -> ATTACKER:MAC:ADDR",
        "4. Victim sends ALL traffic destined for gateway to attacker",
        "5. Attacker forwards to real gateway (MitM) or drops (DoS)",
        "6. Attacker can read, modify, or inject data",
    ]
    for step in steps:
        print(f"    {step}")


def exercise_2():
    """
    Problem 2: DoS/DDoS
    - DoS vs DDoS.
    - SYN Flood principle and countermeasures.
    """
    print("DoS vs DDoS:")
    print(f"\n  {'':5s} {'DoS':35s} {'DDoS':35s}")
    print(f"  {'-'*75}")
    comparison = [
        ("Source", "Single attacker", "Many compromised machines (botnet)"),
        ("Scale", "Limited by one machine", "Massive (millions of bots possible)"),
        ("Detection", "Easier (single source IP)", "Harder (distributed sources)"),
        ("Mitigation", "Block source IP", "Rate limiting, CDN, scrubbing centers"),
    ]
    for aspect, dos, ddos in comparison:
        print(f"  {aspect:10s} {dos:35s} {ddos:35s}")

    print("\n\nSYN Flood Attack:")
    print("  Principle:")
    print("    1. Attacker sends massive SYN packets with spoofed source IPs")
    print("    2. Server allocates resources for each half-open connection")
    print("    3. Server sends SYN-ACK to spoofed addresses (no reply comes)")
    print("    4. Server's SYN queue fills up")
    print("    5. Legitimate clients cannot establish new connections")

    print("\n  Countermeasures:")
    countermeasures = [
        ("SYN Cookies", "Server encodes state in the SYN-ACK sequence number; "
         "no state stored until 3-way handshake completes"),
        ("Rate Limiting", "Limit SYN packets per source IP per second"),
        ("Firewall Rules", "Drop SYN packets from known-bad sources"),
        ("TCP timeout reduction", "Reduce half-open connection timeout"),
        ("Overprovisioning", "Increase SYN queue size as temporary measure"),
    ]
    for name, detail in countermeasures:
        print(f"    - {name}: {detail}")


def exercise_3():
    """
    Problem 3: Web Security
    - Most effective method to prevent SQL Injection.
    - Stored XSS vs Reflected XSS.
    """
    print("SQL Injection Prevention:")
    print("  Most effective: Parameterized Queries (Prepared Statements)")
    print()
    print("  Vulnerable (string concatenation):")
    print("    query = f\"SELECT * FROM users WHERE id = '{user_input}'\"")
    print("    Input: ' OR '1'='1")
    print("    Result: SELECT * FROM users WHERE id = '' OR '1'='1'  (returns all)")
    print()
    print("  Secure (parameterized):")
    print("    query = \"SELECT * FROM users WHERE id = ?\"")
    print("    cursor.execute(query, (user_input,))")
    print("    Input treated as DATA, never as SQL code")

    print("\n\nStored XSS vs Reflected XSS:")
    print("\n  Stored XSS:")
    print("    - Malicious script saved to server (DB, forum post, comment)")
    print("    - Executes for every user who views the content")
    print("    - More dangerous: persistent, affects many users")
    print("    - Example: <script>steal(document.cookie)</script> in a blog comment")

    print("\n  Reflected XSS:")
    print("    - Malicious script embedded in URL/request")
    print("    - Server reflects it back in the response")
    print("    - Requires victim to click crafted link")
    print("    - Example: search?q=<script>alert('xss')</script>")


def exercise_4():
    """
    Problem 4: MITM
    - SSL Stripping attack.
    - How HSTS prevents it.
    """
    print("SSL Stripping Attack:")
    print()
    print("  Normal flow:")
    print("    User -> http://bank.com -> 301 Redirect -> https://bank.com (secure)")
    print()
    print("  SSL Stripping:")
    print("    1. Attacker performs ARP spoofing (becomes MitM)")
    print("    2. User requests http://bank.com")
    print("    3. Attacker intercepts, fetches https://bank.com from server")
    print("    4. Attacker serves HTTP version to user (strips HTTPS)")
    print("    5. User sees http://bank.com (no padlock)")
    print("    6. Attacker reads all traffic in plaintext")

    print("\n  HSTS (HTTP Strict Transport Security) Prevention:")
    print("    Server header: Strict-Transport-Security: max-age=31536000; includeSubDomains")
    print()
    print("    How it works:")
    print("    1. First visit: Browser receives HSTS header over HTTPS")
    print("    2. Browser remembers: 'Always use HTTPS for this domain'")
    print("    3. All future requests automatically upgraded to HTTPS")
    print("    4. Browser refuses HTTP connections to this domain")
    print("    5. SSL stripping fails because browser never sends HTTP")
    print()
    print("    HSTS Preload: Domain added to browser's built-in list")
    print("    Even the first visit is protected (no TOFU problem)")


def exercise_5():
    """
    Problem 5: IDS/IPS
    - Signature-based vs anomaly detection.
    - Why IDS and IPS are deployed at different locations.
    """
    print("IDS/IPS Detection Methods:")

    comparison = [
        ("Approach", "Pattern matching against known threats",
         "Baseline behavior analysis, flags deviations"),
        ("Strengths", "Accurate for known threats, low false positives",
         "Detects zero-day attacks, novel threats"),
        ("Weaknesses", "Cannot detect unknown threats (zero-day)",
         "Higher false positives, needs training period"),
        ("Updates", "Requires frequent signature updates",
         "Self-learning, adapts to environment"),
    ]

    print(f"  {'Aspect':12s} {'Signature-based':40s} {'Anomaly-based':35s}")
    print(f"  {'-'*87}")
    for aspect, sig, anom in comparison:
        print(f"  {aspect:12s} {sig:40s} {anom:35s}")

    print("\n\nIDS vs IPS Deployment:")
    print("\n  IDS (Intrusion Detection System):")
    print("    Placement: Out-of-band (passive tap/span port)")
    print("    Action: Alerts only, does NOT block traffic")
    print("    Impact: Zero impact on network performance")
    print("    Use: Monitoring, forensics, threat intelligence")

    print("\n  IPS (Intrusion Prevention System):")
    print("    Placement: Inline (traffic flows through it)")
    print("    Action: Can actively block malicious traffic")
    print("    Impact: Adds latency; misconfiguration can block legitimate traffic")
    print("    Use: Active threat prevention at network perimeter")


def exercise_6():
    """
    Problem 6: Scenario Analysis
    - ARP table abnormally modified
    - No padlock icon on web server
    - Large volume of SELECT queries on database
    """
    scenarios = [
        {
            "observation": "ARP table abnormally modified in company network",
            "possible_attack": "ARP spoofing / ARP poisoning",
            "countermeasures": [
                "Enable Dynamic ARP Inspection (DAI) on switches",
                "Use static ARP entries for critical servers/gateways",
                "Deploy 802.1X port-based authentication",
                "Monitor ARP tables with arpwatch",
            ],
        },
        {
            "observation": "No padlock icon when accessing web server",
            "possible_attack": "SSL stripping or misconfigured HTTPS",
            "countermeasures": [
                "Enable HSTS on the web server",
                "Force HTTPS redirect at load balancer/reverse proxy",
                "Check certificate validity and chain",
                "Implement HSTS preloading",
            ],
        },
        {
            "observation": "Large volume of SELECT queries on database",
            "possible_attack": "SQL injection (data exfiltration) or brute-force enumeration",
            "countermeasures": [
                "Implement WAF to filter SQL injection patterns",
                "Use parameterized queries in application code",
                "Set database rate limiting and query timeouts",
                "Review application logs for suspicious query patterns",
                "Apply principle of least privilege to DB accounts",
            ],
        },
    ]

    print("Security Scenario Analysis:")
    for s in scenarios:
        print(f"\n  Observation: {s['observation']}")
        print(f"  Possible attack: {s['possible_attack']}")
        print(f"  Countermeasures:")
        for c in s["countermeasures"]:
            print(f"    - {c}")


def exercise_7():
    """
    Problem 7: Comprehensive Security
    Security vulnerabilities in: Internet --- Web Server --- DB Server (same network)
    """
    print("Architecture Security Analysis:")
    print("  Internet --- Web Server --- DB Server")
    print("                    (same network)")

    vulnerabilities = [
        ("No network segmentation",
         "Web and DB on same network. Compromised web server = direct DB access."),
        ("No DMZ",
         "Web server should be in DMZ, DB in internal zone with firewall between."),
        ("No WAF",
         "Web server directly exposed to Internet without application-layer filtering."),
        ("Direct DB exposure",
         "If web server compromised, attacker has network access to DB."),
        ("No internal firewall",
         "No traffic filtering between web tier and data tier."),
    ]

    print("\n  Vulnerabilities:")
    for i, (vuln, detail) in enumerate(vulnerabilities, 1):
        print(f"    {i}. {vuln}: {detail}")

    print("\n  Recommended architecture:")
    print("    Internet -> WAF/LB -> [DMZ: Web Server] -> Internal FW -> [DB Server]")
    print("    - Web server in DMZ (accessible from Internet)")
    print("    - DB server in internal zone (only web server can access)")
    print("    - Firewall rules: only port 5432/3306 from web to DB")


def exercise_8():
    """
    Problem 8: Incident Response
    List ransomware infection response procedures in order.
    """
    print("Ransomware Incident Response Procedures:")

    steps = [
        ("1. Isolate", [
            "Disconnect affected systems from network immediately",
            "Disable Wi-Fi and Bluetooth",
            "Do NOT power off (preserve memory forensics)",
        ]),
        ("2. Identify", [
            "Determine ransomware variant and scope",
            "Identify patient zero (initial infection vector)",
            "Assess what data/systems are affected",
        ]),
        ("3. Notify", [
            "Alert security team and management",
            "Report to law enforcement if required",
            "Notify regulatory bodies (GDPR, HIPAA) if data breach",
        ]),
        ("4. Contain", [
            "Block C2 (command and control) server communications",
            "Revoke compromised credentials",
            "Apply emergency patches if vulnerability is known",
        ]),
        ("5. Eradicate", [
            "Remove malware from all affected systems",
            "Rebuild systems from clean backups if possible",
            "Verify backup integrity before restoring",
        ]),
        ("6. Recover", [
            "Restore data from verified clean backups",
            "Bring systems back online gradually",
            "Monitor closely for re-infection",
        ]),
        ("7. Post-Incident", [
            "Conduct lessons-learned review",
            "Update security policies and procedures",
            "Improve backup strategy and test regularly",
            "Enhance employee security awareness training",
        ]),
    ]

    for phase, actions in steps:
        print(f"\n  {phase}:")
        for action in actions:
            print(f"    - {action}")

    print("\n  Critical: Do NOT pay the ransom (no guarantee of recovery,")
    print("  funds further criminal activity, may violate sanctions).")


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
