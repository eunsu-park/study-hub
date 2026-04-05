# Network Scanning

**Previous**: [02. Reconnaissance](./02_Reconnaissance.md) | **Next**: [04. Vulnerability Assessment](./04_Vulnerability_Assessment.md)

---

Network scanning is the systematic process of discovering hosts, open ports, running services, and operating systems on a target network. While reconnaissance gathers publicly available information, network scanning actively probes the target infrastructure to build a detailed map of the attack surface.

> **IMPORTANT**: All techniques described in this lesson must only be used against systems you own or have explicit written authorization to test. Unauthorized scanning is a criminal offense.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Perform host discovery using ARP, ICMP, and TCP techniques
2. Understand TCP three-way handshake and how SYN scans work
3. Use Nmap for comprehensive network scanning
4. Detect services and versions running on open ports
5. Fingerprint operating systems remotely
6. Evade firewalls and intrusion detection systems
7. Automate scanning workflows with Python and nmap libraries
8. Interpret scan results for penetration test planning

---

## Table of Contents

1. [Host Discovery Techniques](#1-host-discovery-techniques)
2. [TCP and UDP Port Scanning](#2-tcp-and-udp-port-scanning)
3. [Nmap Deep Dive](#3-nmap-deep-dive)
4. [Service and Version Detection](#4-service-and-version-detection)
5. [OS Fingerprinting](#5-os-fingerprinting)
6. [Firewall and IDS Evasion](#6-firewall-and-ids-evasion)
7. [Masscan for Speed](#7-masscan-for-speed)
8. [Network Mapping and Visualization](#8-network-mapping-and-visualization)
9. [Scanning Automation with Python](#9-scanning-automation-with-python)
10. [Countermeasures and Detection](#10-countermeasures-and-detection)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Host Discovery Techniques

Host discovery determines which IP addresses in a target range have active hosts. This is the first step — before scanning ports, we need to know which hosts exist.

### 1.1 ARP Discovery (Layer 2)

ARP discovery is the fastest and most reliable method on local networks. ARP requests cannot be blocked by host firewalls because ARP operates below the IP layer.

```bash
# ARP scan — most reliable on local networks
nmap -sn -PR 192.168.1.0/24

# Using arp-scan
arp-scan --localnet

# Nmap ARP ping (default for local networks)
nmap -sn 192.168.1.0/24
```

### 1.2 ICMP Discovery (Layer 3)

```bash
# ICMP echo (traditional ping sweep)
nmap -sn -PE 10.0.0.0/24

# ICMP timestamp (bypasses some firewalls blocking echo)
nmap -sn -PP 10.0.0.0/24

# ICMP address mask
nmap -sn -PM 10.0.0.0/24
```

### 1.3 TCP/UDP Discovery (Layer 4)

```bash
# TCP SYN ping on common ports
nmap -sn -PS22,80,443 10.0.0.0/24

# TCP ACK ping (bypasses stateless firewalls)
nmap -sn -PA80,443 10.0.0.0/24

# UDP ping
nmap -sn -PU53,161 10.0.0.0/24
```

```python
"""
Host discovery module for authorized network assessment.

Demonstrates how network scanners discover live hosts
using TCP probes at the socket level.
"""

import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import ipaddress


@dataclass
class HostResult:
    """Result of a host discovery probe."""
    ip: str
    is_alive: bool
    method: str
    response_time_ms: Optional[float] = None


def tcp_probe(target: str, port: int, timeout: float = 1.5) -> bool:
    """Check if a host is alive using TCP connection attempt."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((target, port))
        sock.close()
        return result == 0 or result == 111  # open or RST = alive
    except (socket.timeout, OSError):
        return False


def discover_hosts(
    network: str,
    ports: list[int] = None,
    max_workers: int = 50,
    timeout: float = 1.5,
) -> list[HostResult]:
    """
    Discover live hosts in a CIDR range using TCP probes.

    Args:
        network: CIDR notation (e.g., "192.168.1.0/24")
        ports: Ports to probe (default: [80, 443, 22])
        max_workers: Concurrent threads
        timeout: Socket timeout in seconds
    """
    if ports is None:
        ports = [80, 443, 22]

    net = ipaddress.ip_network(network, strict=False)
    results = []

    def check_host(ip_str: str) -> Optional[HostResult]:
        for port in ports:
            if tcp_probe(ip_str, port, timeout):
                return HostResult(
                    ip=ip_str, is_alive=True,
                    method=f"TCP/{port}",
                )
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_host, str(ip)): str(ip)
            for ip in net.hosts()
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return sorted(results, key=lambda r: ipaddress.ip_address(r.ip))


if __name__ == "__main__":
    print("Host Discovery Module")
    print("=" * 40)
    print("Only use against authorized targets.")
    print("\nExample:")
    print("  hosts = discover_hosts('192.168.1.0/24')")
    print("  for h in hosts:")
    print("      print(f'{h.ip} is alive ({h.method})')")
```

---

## 2. TCP and UDP Port Scanning

### 2.1 TCP Three-Way Handshake

Understanding the TCP handshake is fundamental to understanding scan types:

```
Client          Server
  │──── SYN ──────▶│   Step 1: Client initiates
  │◀── SYN/ACK ───│   Step 2: Server acknowledges
  │──── ACK ──────▶│   Step 3: Connection established
```

### 2.2 Scan Types

**TCP SYN Scan** (Half-open scan): Sends SYN, receives SYN/ACK (open) or RST (closed). Does not complete the handshake — stealthier.

**TCP Connect Scan**: Completes the full three-way handshake. More detectable but works without root privileges.

**UDP Scan**: Sends UDP packets. Open ports may not respond; closed ports send ICMP Port Unreachable.

```python
"""
TCP port scanner implementation for educational purposes.

Demonstrates how port scanners work at the socket level.
For real assessments, use Nmap or Masscan.
"""

import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class PortResult:
    """Result of scanning a single port."""
    port: int
    state: str  # open, closed, filtered
    service: str = ""
    banner: str = ""
    response_time_ms: float = 0.0


SERVICES = {
    21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 80: "HTTP", 110: "POP3", 135: "MSRPC",
    139: "NetBIOS", 143: "IMAP", 443: "HTTPS", 445: "SMB",
    993: "IMAPS", 995: "POP3S", 1433: "MSSQL", 3306: "MySQL",
    3389: "RDP", 5432: "PostgreSQL", 5900: "VNC", 6379: "Redis",
    8080: "HTTP-Proxy", 8443: "HTTPS-Alt", 27017: "MongoDB",
}


def scan_port(host: str, port: int, timeout: float = 2.0) -> PortResult:
    """Scan a single TCP port using connect scan."""
    start = time.monotonic()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        elapsed = (time.monotonic() - start) * 1000

        if result == 0:
            banner = ""
            try:
                sock.send(b"\r\n")
                banner = sock.recv(1024).decode("utf-8", errors="ignore").strip()
            except (socket.timeout, OSError):
                pass
            sock.close()
            return PortResult(
                port=port, state="open",
                service=SERVICES.get(port, "unknown"),
                banner=banner[:200],
                response_time_ms=round(elapsed, 2),
            )
        sock.close()
        return PortResult(port=port, state="closed")
    except socket.timeout:
        return PortResult(port=port, state="filtered")
    except OSError:
        return PortResult(port=port, state="error")


def scan_host(
    host: str,
    ports: list[int],
    max_workers: int = 100,
) -> list[PortResult]:
    """Scan multiple ports on a host concurrently."""
    open_ports = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scan_port, host, port): port
            for port in ports
        }
        for future in as_completed(futures):
            result = future.result()
            if result.state == "open":
                open_ports.append(result)
    return sorted(open_ports, key=lambda r: r.port)


def report(host: str, results: list[PortResult]) -> str:
    """Generate a formatted scan report."""
    lines = [
        f"Scan Report: {host}",
        f"Open ports: {len(results)}",
        "=" * 60,
        f"{'PORT':>8}  {'STATE':8}  {'SERVICE':15}  BANNER",
        "-" * 60,
    ]
    for r in results:
        lines.append(
            f"{r.port:>8}  {r.state:8}  {r.service:15}  {r.banner[:40]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("TCP Port Scanner (Educational)")
    print("=" * 40)
    print(f"Known services: {len(SERVICES)}")
    print("Usage: results = scan_host('target', [22, 80, 443])")
```

---

## 3. Nmap Deep Dive

Nmap is the industry-standard network scanner. Mastering Nmap is essential for any penetration tester.

### 3.1 Essential Nmap Commands

```bash
# Basic SYN scan with service detection
nmap -sS -sV -O 10.0.0.1

# Comprehensive scan (SYN + version + scripts + OS)
nmap -sS -sV -sC -O -p- 10.0.0.1

# Fast scan (top 100 ports)
nmap -F 10.0.0.1

# Aggressive scan (version, scripts, OS, traceroute)
nmap -A 10.0.0.1

# Scan entire subnet, output all formats
nmap -sS -sV -O -oA scan_results 10.0.0.0/24

# Scan specific ports
nmap -p 80,443,8080,8443 10.0.0.1

# Scan all 65535 ports
nmap -p- 10.0.0.1

# UDP scan (slow but important)
nmap -sU --top-ports 100 10.0.0.1
```

### 3.2 Nmap Scripting Engine (NSE)

NSE extends Nmap with powerful scripting capabilities:

```bash
# Run default scripts
nmap -sC 10.0.0.1

# Run specific script category
nmap --script vuln 10.0.0.1
nmap --script "http-*" 10.0.0.1

# Run specific script
nmap --script http-title 10.0.0.1
nmap --script ssl-heartbleed 10.0.0.1

# Script categories: auth, broadcast, brute, default,
# discovery, dos, exploit, external, fuzzer, intrusive,
# malware, safe, version, vuln
```

### 3.3 Output Formats

```bash
# Normal output
nmap -oN output.txt 10.0.0.1

# XML output (for parsing)
nmap -oX output.xml 10.0.0.1

# Grepable output
nmap -oG output.gnmap 10.0.0.1

# All formats at once
nmap -oA output_base 10.0.0.1
```

---

## 4. Service and Version Detection

Service detection identifies the specific software and version running on each open port.

```bash
# Version detection (intensity 0-9, default 7)
nmap -sV 10.0.0.1

# Aggressive version detection
nmap -sV --version-intensity 9 10.0.0.1

# Light version detection (faster)
nmap -sV --version-light 10.0.0.1
```

### 4.1 Banner Grabbing

```python
"""
Service banner grabbing module.

Connects to open ports and captures service banners
for version identification.
"""

import socket
from dataclasses import dataclass


@dataclass
class BannerResult:
    """Captured service banner."""
    host: str
    port: int
    banner: str
    service_guess: str = ""


# Protocol-specific probes
PROBES = {
    "http": b"HEAD / HTTP/1.1\r\nHost: {host}\r\n\r\n",
    "smtp": b"EHLO test\r\n",
    "ftp": b"",  # FTP sends banner on connect
    "ssh": b"",  # SSH sends banner on connect
    "generic": b"\r\n\r\n",
}


def grab_banner(
    host: str,
    port: int,
    timeout: float = 5.0,
    probe: str = "generic",
) -> BannerResult:
    """Grab a service banner from an open port."""
    result = BannerResult(host=host, port=port, banner="")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        # Some services send banner immediately
        try:
            initial = sock.recv(1024)
            if initial:
                result.banner = initial.decode("utf-8", errors="replace").strip()
        except socket.timeout:
            pass

        # Send probe if no initial banner
        if not result.banner and probe in PROBES:
            probe_data = PROBES[probe]
            if b"{host}" in probe_data:
                probe_data = probe_data.replace(b"{host}", host.encode())
            if probe_data:
                sock.send(probe_data)
                try:
                    response = sock.recv(4096)
                    result.banner = response.decode("utf-8", errors="replace").strip()
                except socket.timeout:
                    pass

        sock.close()

        # Guess service from banner
        banner_lower = result.banner.lower()
        if "ssh" in banner_lower:
            result.service_guess = "SSH"
        elif "http" in banner_lower:
            result.service_guess = "HTTP"
        elif "smtp" in banner_lower:
            result.service_guess = "SMTP"
        elif "ftp" in banner_lower:
            result.service_guess = "FTP"
        elif "mysql" in banner_lower:
            result.service_guess = "MySQL"

    except (socket.timeout, ConnectionRefusedError, OSError):
        pass

    return result


if __name__ == "__main__":
    print("Banner Grabbing Module")
    print("=" * 40)
    print("Usage: result = grab_banner('target', 22)")
    print("Only use against authorized targets.")
```

---

## 5. OS Fingerprinting

OS fingerprinting determines the operating system of a target host by analyzing network protocol behavior.

### 5.1 Active OS Fingerprinting

```bash
# Nmap OS detection (requires root)
nmap -O 10.0.0.1

# Aggressive OS detection
nmap -O --osscan-guess 10.0.0.1

# Combined with version detection
nmap -O -sV 10.0.0.1
```

### 5.2 Passive OS Fingerprinting

Passive fingerprinting analyzes normal traffic without sending special probes:

- **TTL values**: Linux default 64, Windows default 128, Cisco default 255
- **TCP window size**: Varies by OS and version
- **TCP options**: Order and values differ between implementations
- **MSS values**: Reveal MTU and OS characteristics

---

## 6. Firewall and IDS Evasion

### 6.1 Nmap Evasion Techniques

```bash
# Fragment packets
nmap -f 10.0.0.1

# Set specific MTU
nmap --mtu 24 10.0.0.1

# Use decoy addresses
nmap -D RND:10 10.0.0.1

# Spoof source port (trusted port)
nmap --source-port 53 10.0.0.1

# Randomize target order
nmap --randomize-hosts 10.0.0.0/24

# Slow scan timing
nmap -T0 10.0.0.1  # Paranoid (5 min between probes)
nmap -T1 10.0.0.1  # Sneaky (15 sec between probes)

# Append random data
nmap --data-length 25 10.0.0.1
```

### 6.2 Timing Templates

| Template | Name | Use Case |
|----------|------|----------|
| `-T0` | Paranoid | IDS evasion, very slow |
| `-T1` | Sneaky | IDS evasion, slow |
| `-T2` | Polite | Reduced bandwidth |
| `-T3` | Normal | Default |
| `-T4` | Aggressive | Fast, reliable networks |
| `-T5` | Insane | Very fast, may miss results |

---

## 7. Masscan for Speed

Masscan can scan the entire internet in under 6 minutes. It's the fastest port scanner available, using asynchronous SYN scanning.

```bash
# Fast scan of common ports
masscan 10.0.0.0/24 -p 80,443,8080 --rate 1000

# Scan all ports at high speed
masscan 10.0.0.0/24 -p 1-65535 --rate 10000

# Output in Nmap-compatible XML
masscan 10.0.0.0/24 -p 1-65535 --rate 5000 -oX masscan_output.xml

# Then use Nmap for detailed service detection on discovered ports
nmap -sV -sC -p 80,443,8080 -iL masscan_hosts.txt
```

---

## 8. Network Mapping and Visualization

```python
"""
Network scan results parser and analyzer.

Parses Nmap XML output and generates structured reports
for penetration test planning.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Service:
    """A network service discovered on a port."""
    port: int
    protocol: str
    state: str
    service_name: str = ""
    product: str = ""
    version: str = ""
    extra_info: str = ""

    @property
    def display(self) -> str:
        parts = [f"{self.port}/{self.protocol}"]
        if self.service_name:
            parts.append(self.service_name)
        if self.product:
            parts.append(self.product)
        if self.version:
            parts.append(self.version)
        return " | ".join(parts)


@dataclass
class Host:
    """A discovered network host."""
    ip: str
    hostname: str = ""
    os_guess: str = ""
    state: str = "up"
    services: list[Service] = field(default_factory=list)

    @property
    def open_ports(self) -> list[int]:
        return [s.port for s in self.services if s.state == "open"]


def parse_nmap_xml(xml_path: str) -> list[Host]:
    """Parse Nmap XML output into structured Host objects."""
    hosts = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for host_elem in root.findall("host"):
            # Get IP address
            addr_elem = host_elem.find("address[@addrtype='ipv4']")
            if addr_elem is None:
                continue

            host = Host(ip=addr_elem.get("addr", ""))

            # Hostname
            hostname_elem = host_elem.find(".//hostname")
            if hostname_elem is not None:
                host.hostname = hostname_elem.get("name", "")

            # OS detection
            osmatch = host_elem.find(".//osmatch")
            if osmatch is not None:
                host.os_guess = osmatch.get("name", "")

            # Ports/services
            for port_elem in host_elem.findall(".//port"):
                state_elem = port_elem.find("state")
                service_elem = port_elem.find("service")

                service = Service(
                    port=int(port_elem.get("portid", 0)),
                    protocol=port_elem.get("protocol", "tcp"),
                    state=state_elem.get("state", "") if state_elem is not None else "",
                )

                if service_elem is not None:
                    service.service_name = service_elem.get("name", "")
                    service.product = service_elem.get("product", "")
                    service.version = service_elem.get("version", "")

                if service.state == "open":
                    host.services.append(service)

            if host.services:
                hosts.append(host)

    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error parsing XML: {e}")

    return hosts


def generate_target_report(hosts: list[Host]) -> str:
    """Generate a penetration test target report from scan results."""
    lines = [
        "Network Scan Analysis Report",
        "=" * 60,
        f"Total hosts discovered: {len(hosts)}",
        f"Total open ports: {sum(len(h.services) for h in hosts)}",
        "",
    ]

    # Group by interesting services
    web_servers = []
    databases = []
    remote_access = []

    for host in hosts:
        for svc in host.services:
            if svc.port in (80, 443, 8080, 8443) or "http" in svc.service_name:
                web_servers.append((host, svc))
            elif svc.port in (3306, 5432, 1433, 27017, 6379):
                databases.append((host, svc))
            elif svc.port in (22, 3389, 5900, 23):
                remote_access.append((host, svc))

    lines.append(f"Web servers: {len(web_servers)}")
    lines.append(f"Databases: {len(databases)}")
    lines.append(f"Remote access: {len(remote_access)}")
    lines.append("")

    for host in hosts:
        lines.append(f"\n--- {host.ip} ({host.hostname or 'no hostname'}) ---")
        if host.os_guess:
            lines.append(f"  OS: {host.os_guess}")
        for svc in host.services:
            lines.append(f"  {svc.display}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Nmap XML Parser")
    print("=" * 40)
    print("Usage:")
    print("  1. Run: nmap -sV -oX scan.xml target")
    print("  2. Parse: hosts = parse_nmap_xml('scan.xml')")
    print("  3. Report: print(generate_target_report(hosts))")
```

---

## 9. Scanning Automation with Python

```python
"""
Automated scanning pipeline that combines multiple tools.

Orchestrates host discovery, port scanning, and service
detection into a single workflow.
"""

import json
import subprocess
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ScanConfig:
    """Configuration for an automated scan."""
    target: str
    output_dir: str = "./scan_results"
    scan_type: str = "standard"  # quick, standard, comprehensive
    max_rate: int = 1000
    timing: int = 4  # Nmap timing template (0-5)

    @property
    def nmap_args(self) -> list[str]:
        """Generate Nmap arguments based on scan type."""
        base = ["-sS", "-sV", f"-T{self.timing}"]
        if self.scan_type == "quick":
            base.extend(["-F", "--top-ports", "100"])
        elif self.scan_type == "standard":
            base.extend(["--top-ports", "1000", "-sC"])
        elif self.scan_type == "comprehensive":
            base.extend(["-p-", "-sC", "-O", "--script", "vuln"])
        return base


@dataclass
class ScanResult:
    """Complete scan results."""
    config: ScanConfig
    start_time: str = ""
    end_time: str = ""
    hosts_discovered: int = 0
    total_open_ports: int = 0
    findings: list[dict] = field(default_factory=list)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


class ScanPipeline:
    """Automated scanning pipeline."""

    def __init__(self, config: ScanConfig):
        self.config = config
        self.result = ScanResult(config=config)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def check_tools(self) -> dict[str, bool]:
        """Verify required tools are installed."""
        tools = ["nmap", "masscan"]
        return {t: shutil.which(t) is not None for t in tools}

    def run_nmap(self) -> Optional[str]:
        """Execute Nmap scan."""
        output_base = f"{self.config.output_dir}/nmap_scan"
        cmd = [
            "nmap",
            *self.config.nmap_args,
            "-oA", output_base,
            self.config.target,
        ]
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            return f"{output_base}.xml"
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Nmap error: {e}")
            return None

    def run(self) -> ScanResult:
        """Execute the full scanning pipeline."""
        self.result.start_time = datetime.utcnow().isoformat()

        # Check tools
        tools = self.check_tools()
        missing = [t for t, available in tools.items() if not available]
        if "nmap" in missing:
            print("[ERROR] Nmap is required but not installed")
            return self.result

        # Run Nmap
        print(f"\n[*] Starting {self.config.scan_type} scan of {self.config.target}")
        xml_path = self.run_nmap()

        self.result.end_time = datetime.utcnow().isoformat()

        # Save results
        result_path = f"{self.config.output_dir}/scan_summary.json"
        self.result.save(result_path)
        print(f"\n[*] Results saved to {result_path}")

        return self.result


if __name__ == "__main__":
    print("Automated Scanning Pipeline")
    print("=" * 40)
    print("Usage:")
    print("  config = ScanConfig(target='10.0.0.0/24', scan_type='standard')")
    print("  pipeline = ScanPipeline(config)")
    print("  result = pipeline.run()")
    print("\nOnly use against authorized targets.")
```

---

## 10. Countermeasures and Detection

### 10.1 Detecting Port Scans

| Scan Type | Detection Indicators |
|-----------|---------------------|
| SYN scan | Many SYN packets without completing handshakes |
| Connect scan | Many short-lived connections |
| UDP scan | ICMP unreachable messages |
| Masscan | High-rate SYN packets from single source |
| OS fingerprint | Unusual TCP flag combinations |

### 10.2 Defense Strategies

- **Firewall rules**: Block unnecessary inbound ports
- **IDS/IPS**: Configure rules for scan detection (Snort, Suricata)
- **Rate limiting**: Limit connection rates per source IP
- **Port knocking**: Require specific sequence before opening ports
- **Honeypots**: Deploy decoy services to detect scanning

---

## 11. Exercises

1. **Host Discovery**: Set up a lab network with 5+ VMs and use three different host discovery methods. Compare the results.

2. **Port Scanning**: Scan a Metasploitable VM with SYN, Connect, and UDP scans. Document the differences in results and detection.

3. **Nmap Mastery**: Perform a comprehensive Nmap scan against a lab target. Use NSE scripts to gather additional information.

4. **Evasion Testing**: Scan a target through a firewall using various evasion techniques. Test which methods successfully bypass the firewall.

5. **Automation**: Extend the ScanPipeline class to include Masscan for initial port discovery followed by Nmap for service detection.

6. **Analysis**: Parse an Nmap XML output file and create a prioritized target list for a penetration test.

---

## 12. Summary

Network scanning builds on reconnaissance to create a detailed map of the target's attack surface:

- **Host discovery** identifies live systems using ARP, ICMP, and TCP techniques
- **Port scanning** reveals open services — SYN scans are the standard for stealth
- **Nmap** is the essential tool — mastering its options and NSE scripts is critical
- **Service detection** identifies specific software versions for vulnerability matching
- **OS fingerprinting** reveals the target operating system for targeted exploitation
- **Evasion techniques** help bypass firewalls and IDS during authorized assessments
- **Automation** enables consistent, comprehensive scanning across large networks

---

## 13. References

- Nmap Official Documentation: https://nmap.org/book/
- Nmap NSE Script Library: https://nmap.org/nsedoc/
- Masscan: https://github.com/robertdavidgraham/masscan
- RustScan: https://github.com/RustScan/RustScan
- TCP/IP Illustrated, Volume 1 (Stevens)
- Nmap Network Scanning (Fyodor): https://nmap.org/book/
