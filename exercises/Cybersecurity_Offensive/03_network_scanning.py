"""
Exercises for Lesson 03: Network Scanning
Topic: Cybersecurity_Offensive

Practice problems covering port scanning techniques, service detection,
and scan result analysis.
"""


# === Exercise 1: TCP Three-Way Handshake Simulator ===
# Problem: Simulate the TCP three-way handshake states.
# Given a scan type ("connect", "syn", "fin"), return the sequence
# of packets exchanged as a list of strings.

def exercise_1():
    """
    Return packet sequences for each scan type:
      "connect" -> full handshake: ["SYN->", "<-SYN-ACK", "ACK->", ...]
      "syn"     -> half-open: ["SYN->", "<-SYN-ACK", "RST->"]
      "fin"     -> stealth: ["FIN->", ...]
    Also indicate if port is open/closed based on response.
    """
    # TODO: Implement packet sequence for each scan type
    pass


# === Exercise 2: Port Range Parser ===
# Problem: Parse nmap-style port specifications into a list of port numbers.
# Input formats: "80" -> [80], "1-5" -> [1,2,3,4,5],
#                "22,80,443" -> [22,80,443], "1-3,80,8000-8002" -> [1,2,3,80,8000,8001,8002]

def exercise_2():
    """
    Implement parse_port_range(spec: str) -> list[int].
    Test with: "22,80,443,8080-8085,9090"
    Return the sorted list of ports.
    """
    # TODO: Parse the port specification string
    pass


# === Exercise 3: Service Banner Analysis ===
# Problem: Given raw service banners, identify the service and version.
# Return a list of dicts with keys: port, service, version, os_hint.

def exercise_3():
    """
    banners = {
        22: "SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.6",
        80: "Apache/2.4.52 (Ubuntu)",
        3306: "5.7.42-0ubuntu0.18.04.1",
        8080: "Jetty(9.4.51.v20230217)",
    }
    Parse each banner and return list of analysis dicts.
    """
    # TODO: Analyze service banners
    pass


# === Exercise 4: Scan Result Prioritizer ===
# Problem: Given a list of open ports with services, prioritize them
# for further testing. High-priority: known-vulnerable services,
# admin interfaces, databases. Return sorted by priority (high first).

def exercise_4():
    """
    scan_results = [
        {"port": 22, "service": "ssh", "version": "OpenSSH 7.2"},
        {"port": 80, "service": "http", "version": "Apache 2.4.49"},
        {"port": 443, "service": "https", "version": "nginx 1.18"},
        {"port": 3306, "service": "mysql", "version": "MySQL 5.7"},
        {"port": 8080, "service": "http-proxy", "version": "Tomcat 8.5.50"},
        {"port": 6379, "service": "redis", "version": "Redis 6.0.16"},
    ]
    Assign priority (high/medium/low) and return sorted results.
    Hint: Apache 2.4.49 has path traversal CVE, Redis often lacks auth.
    """
    # TODO: Prioritize scan results for exploitation
    pass


if __name__ == "__main__":
    print("=== Exercise 1: TCP Handshake Simulator ===")
    print(exercise_1())
    print("\n=== Exercise 2: Port Range Parser ===")
    print(exercise_2())
    print("\n=== Exercise 3: Service Banner Analysis ===")
    print(exercise_3())
    print("\n=== Exercise 4: Scan Result Prioritizer ===")
    print(exercise_4())
