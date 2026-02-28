"""
Exercises for Lesson 12: DNS
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: Domain Structure Analysis
    Identify each part of:
    a) www.shop.amazon.co.uk
    b) mail.google.com
    c) api.v2.example.org
    """
    domains = [
        {
            "domain": "www.shop.amazon.co.uk",
            "parts": {
                "uk": "TLD (ccTLD - United Kingdom country code)",
                "co": "Second-level domain (for companies in UK)",
                "amazon": "Registered domain name",
                "shop": "Subdomain",
                "www": "Subdomain (hostname prefix)",
            },
        },
        {
            "domain": "mail.google.com",
            "parts": {
                "com": "TLD (gTLD - generic top-level domain)",
                "google": "Second-level domain (SLD) / registered domain",
                "mail": "Subdomain (for email service)",
            },
        },
        {
            "domain": "api.v2.example.org",
            "parts": {
                "org": "TLD (gTLD - for organizations)",
                "example": "Second-level domain (SLD)",
                "v2": "Subdomain (API version)",
                "api": "Subdomain (service type)",
            },
        },
    ]

    print("Domain Structure Analysis:")
    for entry in domains:
        print(f"\n  {entry['domain']}:")
        for part, description in entry["parts"].items():
            print(f"    {part:10s} : {description}")


def exercise_2():
    """
    Problem 2: DNS Record Matching
    a) Web server IPv4 address -> A Record
    b) Mail server -> MX Record
    c) Redirect www to base domain -> CNAME Record
    d) Domain ownership -> TXT Record
    e) Name server -> NS Record
    f) IPv6 address -> AAAA Record
    """
    records = {
        "A":     {"purpose": "Maps hostname to IPv4 address", "example": "example.com -> 93.184.216.34"},
        "AAAA":  {"purpose": "Maps hostname to IPv6 address", "example": "example.com -> 2606:2800:220:1:..."},
        "CNAME": {"purpose": "Alias (redirects one name to another)", "example": "www.example.com -> example.com"},
        "MX":    {"purpose": "Specifies mail server with priority", "example": "example.com -> 10 mail.example.com"},
        "NS":    {"purpose": "Delegates to authoritative name server", "example": "example.com -> ns1.example.com"},
        "TXT":   {"purpose": "Text records (SPF, DKIM, verification)", "example": "v=spf1 include:_spf.google.com"},
        "SOA":   {"purpose": "Start of Authority (zone metadata)", "example": "Primary NS, admin email, serial"},
        "PTR":   {"purpose": "Reverse DNS (IP to hostname)", "example": "34.216.184.93 -> example.com"},
        "SRV":   {"purpose": "Service location (port + host)", "example": "_sip._tcp.example.com -> 5060 sip.example.com"},
    }

    matching = [
        ("Specify web server's IPv4 address", "A"),
        ("Specify mail server", "MX"),
        ("Redirect www to base domain", "CNAME"),
        ("Domain ownership authentication", "TXT"),
        ("Specify name server", "NS"),
        ("Specify IPv6 address", "AAAA"),
    ]

    print("DNS Record Type Matching:")
    for description, record in matching:
        info = records[record]
        print(f"\n  {description}")
        print(f"    -> {record} Record: {info['purpose']}")
        print(f"       Example: {info['example']}")


def exercise_3():
    """
    Problem 3: dig Output Analysis
    ;; ANSWER SECTION:
    example.com.  600  IN  MX  10 mail1.example.com.
    example.com.  600  IN  MX  20 mail2.example.com.
    example.com.  600  IN  MX  30 backup.mail.com.
    """
    mx_records = [
        {"domain": "example.com", "ttl": 600, "priority": 10, "server": "mail1.example.com"},
        {"domain": "example.com", "ttl": 600, "priority": 20, "server": "mail2.example.com"},
        {"domain": "example.com", "ttl": 600, "priority": 30, "server": "backup.mail.com"},
    ]

    print("dig Output Analysis:")
    print("  ;; ANSWER SECTION:")
    for mx in mx_records:
        print(f"  {mx['domain']}.  {mx['ttl']}  IN  MX  {mx['priority']} {mx['server']}.")

    print(f"\n  a) TTL = {mx_records[0]['ttl']} seconds ({mx_records[0]['ttl'] // 60} minutes)")
    print(f"     DNS caches will hold this record for 10 minutes before re-querying.")

    # Sort by priority (lowest = highest priority)
    sorted_mx = sorted(mx_records, key=lambda x: x["priority"])
    print(f"\n  b) Primary mail server: {sorted_mx[0]['server']} (priority {sorted_mx[0]['priority']})")
    print(f"     Lower priority number = tried first.")
    print(f"     Failover order:")
    for mx in sorted_mx:
        print(f"       Priority {mx['priority']}: {mx['server']}")

    print(f"\n  c) If all mail servers are down:")
    print(f"     Mail delivery FAILS. The sending MTA will:")
    print(f"     - Retry for a configurable period (typically 4-5 days)")
    print(f"     - Eventually generate a bounce (Non-Delivery Report) to the sender")


def exercise_4():
    """
    Problem 4: DNS Query Practice
    Simulated dig outputs for google.com.

    Reasoning: Understanding DNS query output helps troubleshoot
    name resolution issues and verify DNS configuration.
    """
    print("DNS Query Practice (simulated):")

    # Simulate dig google.com A
    print(f"\n  $ dig google.com A")
    print(f"  ;; ANSWER SECTION:")
    a_records = ["142.250.80.46", "142.250.80.78"]
    for ip in a_records:
        print(f"  google.com.  300  IN  A  {ip}")
    print(f"  ;; Query time: 12 msec")

    # Simulate dig google.com MX
    print(f"\n  $ dig google.com MX")
    print(f"  ;; ANSWER SECTION:")
    mx_servers = [
        (10, "smtp.google.com"),
        (20, "smtp2.google.com"),
        (30, "smtp3.google.com"),
        (40, "smtp4.google.com"),
    ]
    for priority, server in mx_servers:
        print(f"  google.com.  600  IN  MX  {priority} {server}.")

    # Simulate dig +trace
    print(f"\n  $ dig +trace google.com")
    print(f"  ;; Shows the full resolution path:")
    trace_steps = [
        (".", "Root servers (a-m.root-servers.net)"),
        ("com.", "TLD servers (a-m.gtld-servers.net)"),
        ("google.com.", "Authoritative servers (ns1-4.google.com)"),
    ]
    for zone, description in trace_steps:
        print(f"    {zone:20s} -> {description}")

    print(f"\n  Key observations:")
    print(f"    - A records: Multiple IPs for load balancing/redundancy")
    print(f"    - MX records: Multiple servers with priority failover")
    print(f"    - +trace: Shows hierarchical resolution from root")


if __name__ == "__main__":
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
