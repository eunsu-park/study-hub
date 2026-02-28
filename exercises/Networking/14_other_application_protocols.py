"""
Exercises for Lesson 14: Other Application Protocols
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: DHCP
    - Explain each step of DORA.
    - List all information assigned by DHCP server.
    """
    dora_steps = [
        ("Discover", "Client broadcasts: 'Any DHCP servers out there?'",
         "Src: 0.0.0.0:68 -> Dst: 255.255.255.255:67"),
        ("Offer", "Server responds: 'I can give you this IP address'",
         "Server unicasts/broadcasts offer with proposed IP, lease time"),
        ("Request", "Client broadcasts: 'I accept your offer (from server X)'",
         "Broadcast so other DHCP servers know their offer was declined"),
        ("Acknowledge", "Server confirms: 'Done, that IP is yours for N seconds'",
         "Includes all configuration parameters"),
    ]

    print("DHCP DORA Process:")
    for step_name, description, detail in dora_steps:
        print(f"\n  {step_name[0]} - {step_name}:")
        print(f"    {description}")
        print(f"    {detail}")

    print("\n\nInformation assigned by DHCP server:")
    assigned_info = [
        "IP address",
        "Subnet mask",
        "Default gateway (router)",
        "DNS server addresses",
        "Lease duration",
        "Domain name",
        "NTP server (optional)",
        "WINS server (optional, Windows)",
    ]
    for info in assigned_info:
        print(f"  - {info}")


def exercise_2():
    """
    Problem 2: FTP
    - Active vs Passive mode.
    - Security issues and alternatives.
    """
    print("FTP Active vs Passive Mode:")

    print("\n  Active Mode:")
    print("    1. Client connects to server port 21 (control)")
    print("    2. Client sends PORT command with its IP:port")
    print("    3. Server initiates data connection FROM port 20 TO client's port")
    print("    Problem: Server-initiated connection blocked by client's firewall/NAT")

    print("\n  Passive Mode:")
    print("    1. Client connects to server port 21 (control)")
    print("    2. Client sends PASV command")
    print("    3. Server responds with IP:port for data")
    print("    4. Client initiates data connection TO server's specified port")
    print("    Advantage: Both connections initiated by client (NAT/firewall friendly)")

    print("\n\nFTP Security Issues:")
    issues = [
        "Credentials sent in plaintext (username and password)",
        "Data transferred in plaintext (no encryption)",
        "Vulnerable to sniffing and MITM attacks",
        "Active mode exposes client ports to the Internet",
    ]
    for issue in issues:
        print(f"  - {issue}")

    print("\n  Secure Alternatives:")
    alternatives = [
        ("SFTP", "SSH File Transfer Protocol (runs over SSH, port 22)"),
        ("FTPS", "FTP over TLS/SSL (adds encryption to FTP)"),
        ("SCP", "Secure Copy (simple file copy over SSH)"),
    ]
    for name, description in alternatives:
        print(f"  - {name}: {description}")


def exercise_3():
    """
    Problem 3: Email Protocols
    - Roles of SMTP, POP3, and IMAP.
    - POP3 vs IMAP differences.
    """
    protocols = {
        "SMTP": {
            "role": "SENDING email (client to server, server to server)",
            "port": "25 (relay), 587 (submission), 465 (SSL)",
            "analogy": "The postal delivery truck between post offices",
        },
        "POP3": {
            "role": "RECEIVING email (download from server)",
            "port": "110 (plain), 995 (SSL)",
            "analogy": "Taking mail out of your mailbox and bringing it home",
        },
        "IMAP": {
            "role": "RECEIVING email (view/manage on server)",
            "port": "143 (plain), 993 (SSL)",
            "analogy": "Reading mail at the post office without taking it home",
        },
    }

    print("Email Protocol Roles:")
    for name, info in protocols.items():
        print(f"\n  {name}:")
        print(f"    Role: {info['role']}")
        print(f"    Port: {info['port']}")
        print(f"    Analogy: {info['analogy']}")

    print("\n\nPOP3 vs IMAP Comparison:")
    comparison = [
        ("Storage", "Downloads to client, removes from server", "Stays on server"),
        ("Multi-device", "Poor (mail only on one device)", "Excellent (synced across devices)"),
        ("Offline access", "Full (all mail downloaded)", "Limited (depends on cached messages)"),
        ("Server storage", "Minimal (mail removed)", "Requires server storage"),
        ("Bandwidth", "High initial download", "Lower (only headers fetched initially)"),
        ("Folder sync", "No server-side folders", "Full folder structure synced"),
    ]
    print(f"  {'Aspect':18s} {'POP3':35s} {'IMAP':30s}")
    print(f"  {'-'*83}")
    for aspect, pop3, imap in comparison:
        print(f"  {aspect:18s} {pop3:35s} {imap:30s}")


def exercise_4():
    """
    Problem 4: SSH
    - Advantages of public key authentication.
    - Example scenario for SSH local port forwarding.
    """
    print("SSH Public Key Authentication Advantages:")
    advantages = [
        "No password transmission (private key never leaves client)",
        "Immune to brute-force password attacks",
        "Can be automated (scripts, CI/CD) without storing passwords",
        "Stronger security (2048+ bit keys vs typical passwords)",
        "Easy key rotation and revocation",
        "Can restrict key to specific commands (forced command)",
    ]
    for adv in advantages:
        print(f"  - {adv}")

    print("\n\nSSH Local Port Forwarding Example:")
    print("  Scenario: Access internal database through bastion host")
    print()
    print("  Network layout:")
    print("    [Your PC] --> [Bastion (SSH)] --> [DB Server (internal)]")
    print("                  Public IP            10.0.0.50:5432")
    print()
    print("  Command:")
    print("    ssh -L 5432:10.0.0.50:5432 user@bastion.example.com")
    print()
    print("  What happens:")
    print("    1. SSH tunnel established to bastion host")
    print("    2. Local port 5432 on your PC forwards through the tunnel")
    print("    3. Bastion forwards traffic to internal DB at 10.0.0.50:5432")
    print("    4. You connect: psql -h 127.0.0.1 -p 5432 -U dbuser")
    print("    5. Traffic: Your PC -> SSH tunnel -> bastion -> DB (encrypted)")


def exercise_5():
    """
    Problem 5: Port Numbers
    Write port numbers for common protocols.
    """
    port_table = [
        ("DHCP Client", 68, "UDP"),
        ("DHCP Server", 67, "UDP"),
        ("FTP Control", 21, "TCP"),
        ("FTP Data", 20, "TCP"),
        ("SMTP Default", 25, "TCP"),
        ("SMTP SSL", 465, "TCP"),
        ("SMTP Submission", 587, "TCP"),
        ("SSH", 22, "TCP"),
        ("POP3S", 995, "TCP"),
        ("IMAPS", 993, "TCP"),
    ]

    print("Protocol Port Numbers:")
    print(f"  {'Protocol':20s} {'Port':>6s} {'Transport':>10s}")
    print(f"  {'-'*40}")
    for proto, port, transport in port_table:
        print(f"  {proto:20s} {port:>6d} {transport:>10s}")


def exercise_6():
    """
    Problem 6: Practical Problems
    Predict results of SSH tunnel, SFTP upload, and Telnet HTTP test.
    """
    print("Practical Command Analysis:")

    print("\n  1. SSH Tunnel to remote database:")
    print("     $ ssh -L 3306:db.internal:3306 user@bastion.example.com")
    print("     $ mysql -h 127.0.0.1 -P 3306 -u dbuser -p")
    print("     Result: MySQL client connects to local port 3306,")
    print("     which tunnels through bastion to db.internal:3306")
    print("     All traffic is encrypted through the SSH tunnel.")

    print("\n  2. SFTP file upload:")
    print("     $ sftp user@server.example.com")
    print("     sftp> put localfile.txt /home/user/")
    print("     Result: File transferred securely over SSH (port 22).")
    print("     The file appears at /home/user/localfile.txt on the server.")

    print("\n  3. Telnet HTTP test:")
    print("     $ telnet example.com 80")
    print("     GET / HTTP/1.1")
    print("     Host: example.com")
    print("     [blank line]")
    print("     Result: Server returns HTTP response (headers + HTML body).")
    print("     This is plaintext HTTP -- useful for debugging but insecure.")


def exercise_7():
    """
    Problem 7: WebSocket
    - HTTP Polling vs WebSocket.
    - Why does WebSocket handshake through HTTP?
    """
    print("HTTP Polling vs WebSocket:")

    print("\n  HTTP Polling:")
    print("    Client repeatedly asks server: 'Any new data?'")
    print("    Timeline: [Poll]-[Wait]-[Poll]-[Wait]-[Poll]")
    print("    Problems: Wasted requests, latency, server load")

    print("\n  WebSocket:")
    print("    Persistent bidirectional connection after initial handshake")
    print("    Timeline: [Handshake]---[data<->data<->data]---[Close]")
    print("    Benefits: Real-time, low latency, low overhead")

    print("\n  Why WebSocket handshakes through HTTP:")
    reasons = [
        "Compatibility: Works through existing HTTP proxies and firewalls",
        "Port sharing: Uses port 80/443 (no new ports needed)",
        "Infrastructure: Leverages existing web server infrastructure",
        "Upgrade mechanism: HTTP Upgrade header is a standard way to switch protocols",
    ]
    for r in reasons:
        print(f"    - {r}")


def exercise_8():
    """
    Problem 8: Security Comparison
    Security issues of Telnet, FTP, and SMTP, with secure alternatives.
    """
    insecure_protocols = [
        {
            "protocol": "Telnet",
            "port": 23,
            "issues": [
                "All data (including passwords) sent in plaintext",
                "No authentication verification",
                "Vulnerable to sniffing and MITM",
            ],
            "alternative": "SSH (port 22) - encrypted, authenticated",
        },
        {
            "protocol": "FTP",
            "port": "20/21",
            "issues": [
                "Credentials in plaintext",
                "Data transfer unencrypted",
                "Active mode exposes client ports",
            ],
            "alternative": "SFTP (port 22) or FTPS (990) - encrypted transfer",
        },
        {
            "protocol": "SMTP",
            "port": 25,
            "issues": [
                "No encryption by default",
                "No sender authentication (spoofing possible)",
                "Email content readable in transit",
            ],
            "alternative": "SMTP+STARTTLS (587) or SMTPS (465) + SPF/DKIM/DMARC",
        },
    ]

    print("Insecure Protocols and Secure Alternatives:")
    for proto in insecure_protocols:
        print(f"\n  {proto['protocol']} (port {proto['port']}):")
        print(f"    Security issues:")
        for issue in proto["issues"]:
            print(f"      - {issue}")
        print(f"    Secure alternative: {proto['alternative']}")


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
