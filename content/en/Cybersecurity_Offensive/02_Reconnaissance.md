# Reconnaissance

**Previous**: [01. Offensive Security Overview](./01_Offensive_Security_Overview.md) | **Next**: [03. Network Scanning](./03_Network_Scanning.md)

---

Reconnaissance is the first and arguably most critical phase of any penetration test. The quality of information gathered during recon directly determines the effectiveness of subsequent exploitation phases. Professional testers often spend 50-75% of their engagement time on reconnaissance and enumeration, because a well-mapped attack surface reveals the path of least resistance.

> **IMPORTANT**: Perform reconnaissance only against targets you have explicit written authorization to test. Even passive OSINT can have legal implications in some jurisdictions.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Differentiate between passive and active reconnaissance
2. Perform OSINT gathering using multiple sources
3. Enumerate DNS records and discover subdomains
4. Use Google dorking to find exposed sensitive information
5. Leverage Shodan, Censys, and other internet-wide scanners
6. Extract metadata from public documents
7. Map organizational structure and employee information
8. Automate reconnaissance workflows with Python

---

## Table of Contents

1. [Passive vs Active Reconnaissance](#1-passive-vs-active-reconnaissance)
2. [OSINT Fundamentals](#2-osint-fundamentals)
3. [DNS Enumeration](#3-dns-enumeration)
4. [Subdomain Discovery](#4-subdomain-discovery)
5. [Google Dorking](#5-google-dorking)
6. [Shodan and Internet-Wide Scanning](#6-shodan-and-internet-wide-scanning)
7. [WHOIS and Domain Intelligence](#7-whois-and-domain-intelligence)
8. [Metadata Extraction](#8-metadata-extraction)
9. [Social Media and Employee OSINT](#9-social-media-and-employee-osint)
10. [Automated Reconnaissance Framework](#10-automated-reconnaissance-framework)
11. [Countermeasures and Detection](#11-countermeasures-and-detection)
12. [Exercises](#12-exercises)
13. [Summary](#13-summary)
14. [References](#14-references)

---

## 1. Passive vs Active Reconnaissance

Reconnaissance falls into two broad categories based on the level of interaction with the target:

### 1.1 Passive Reconnaissance

Passive recon gathers information without directly interacting with the target's systems. The target has no way to detect that they are being investigated.

**Sources of passive intelligence:**
- Public DNS records and WHOIS databases
- Search engine cached pages and indexed content
- Social media profiles and job postings
- Certificate Transparency logs
- Internet archive (Wayback Machine)
- Code repositories (GitHub, GitLab)
- Shodan, Censys, and other scan databases
- SEC filings, press releases, patent databases

**Advantages**: Undetectable, legal in most jurisdictions, can be done before formal engagement begins.

**Limitations**: Information may be outdated, incomplete, or inaccurate.

### 1.2 Active Reconnaissance

Active recon involves direct interaction with the target — sending packets, making requests, or probing services. The target can potentially detect this activity.

**Active recon techniques:**
- Port scanning and service enumeration
- Web spidering and content discovery
- Banner grabbing and version detection
- DNS zone transfer attempts
- Directory brute-forcing
- Virtual host enumeration

**Advantages**: Current, accurate, comprehensive data.

**Risks**: Detectable by IDS/IPS, may trigger alerts, requires authorization.

```python
"""
Reconnaissance classification framework.

Categorize and track different reconnaissance activities
and their risk levels for engagement planning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ReconType(Enum):
    PASSIVE = "Passive"
    ACTIVE = "Active"
    SEMI_PASSIVE = "Semi-Passive"


class DetectionRisk(Enum):
    NONE = 0        # Cannot be detected
    LOW = 1         # Unlikely to trigger alerts
    MEDIUM = 2      # May appear in logs
    HIGH = 3        # Likely to trigger IDS/IPS alerts
    CERTAIN = 4     # Will definitely be logged


@dataclass
class ReconTechnique:
    """A reconnaissance technique with metadata."""
    name: str
    recon_type: ReconType
    detection_risk: DetectionRisk
    description: str
    tools: list[str] = field(default_factory=list)
    data_gathered: list[str] = field(default_factory=list)
    authorization_required: bool = True
    notes: str = ""


RECON_TECHNIQUES = [
    ReconTechnique(
        name="WHOIS Lookup",
        recon_type=ReconType.PASSIVE,
        detection_risk=DetectionRisk.NONE,
        description="Query public WHOIS databases for domain registration info",
        tools=["whois", "python-whois"],
        data_gathered=["Registrar", "Name servers", "Creation date",
                       "Admin contact (if not redacted)"],
        authorization_required=False,
    ),
    ReconTechnique(
        name="DNS Record Enumeration",
        recon_type=ReconType.SEMI_PASSIVE,
        detection_risk=DetectionRisk.LOW,
        description="Query public DNS servers for various record types",
        tools=["dig", "nslookup", "dnspython"],
        data_gathered=["A records", "MX records", "TXT records",
                       "NS records", "CNAME records"],
        authorization_required=False,
    ),
    ReconTechnique(
        name="Subdomain Brute-Force",
        recon_type=ReconType.ACTIVE,
        detection_risk=DetectionRisk.MEDIUM,
        description="Resolve subdomain wordlists against target DNS",
        tools=["subfinder", "amass", "ffuf", "gobuster"],
        data_gathered=["Hidden subdomains", "Internal hostnames",
                       "Development/staging servers"],
        authorization_required=True,
    ),
    ReconTechnique(
        name="Port Scanning",
        recon_type=ReconType.ACTIVE,
        detection_risk=DetectionRisk.HIGH,
        description="Probe target IP ranges for open ports and services",
        tools=["nmap", "masscan", "rustscan"],
        data_gathered=["Open ports", "Service versions", "OS fingerprint"],
        authorization_required=True,
    ),
    ReconTechnique(
        name="Certificate Transparency",
        recon_type=ReconType.PASSIVE,
        detection_risk=DetectionRisk.NONE,
        description="Search CT logs for certificates issued to target domains",
        tools=["crt.sh", "certspotter", "ctfr"],
        data_gathered=["Subdomains", "Certificate details",
                       "Issuing CAs", "Historical certs"],
        authorization_required=False,
    ),
    ReconTechnique(
        name="Google Dorking",
        recon_type=ReconType.PASSIVE,
        detection_risk=DetectionRisk.NONE,
        description="Advanced search engine queries to find exposed data",
        tools=["Google", "DorkSearch", "GHDB"],
        data_gathered=["Exposed files", "Login pages",
                       "Error messages", "Sensitive directories"],
        authorization_required=False,
        notes="Use responsibly — do not access found resources without auth",
    ),
    ReconTechnique(
        name="Shodan Search",
        recon_type=ReconType.PASSIVE,
        detection_risk=DetectionRisk.NONE,
        description="Query Shodan's database of internet-connected devices",
        tools=["Shodan CLI", "Shodan API", "shodan-python"],
        data_gathered=["Open ports", "Banners", "Vulnerabilities",
                       "SSL certificates", "Screenshots"],
        authorization_required=False,
    ),
    ReconTechnique(
        name="Web Content Discovery",
        recon_type=ReconType.ACTIVE,
        detection_risk=DetectionRisk.HIGH,
        description="Brute-force directories and files on web servers",
        tools=["gobuster", "ffuf", "dirsearch", "feroxbuster"],
        data_gathered=["Hidden directories", "Backup files",
                       "Admin panels", "API endpoints"],
        authorization_required=True,
    ),
]


def plan_recon(
    authorized: bool = True,
    stealth_required: bool = False,
) -> list[ReconTechnique]:
    """
    Select appropriate recon techniques based on constraints.

    Args:
        authorized: Whether formal authorization has been obtained
        stealth_required: Whether detection must be avoided
    """
    techniques = []
    for tech in RECON_TECHNIQUES:
        if not authorized and tech.authorization_required:
            continue
        if stealth_required and tech.detection_risk.value > DetectionRisk.LOW.value:
            continue
        techniques.append(tech)
    return techniques


if __name__ == "__main__":
    print("=== Pre-Authorization Recon (Passive Only) ===\n")
    passive = plan_recon(authorized=False)
    for t in passive:
        print(f"  [{t.recon_type.value:12s}] {t.name}")
        print(f"    Detection: {t.detection_risk.name}")
        print(f"    Tools: {', '.join(t.tools)}")
        print()

    print("\n=== Full Authorized Recon ===\n")
    full = plan_recon(authorized=True)
    for t in full:
        print(f"  [{t.recon_type.value:12s}] {t.name}")
        print(f"    Detection: {t.detection_risk.name}")
        print()
```

---

## 2. OSINT Fundamentals

Open Source Intelligence (OSINT) is the collection and analysis of information from publicly available sources. OSINT forms the backbone of passive reconnaissance.

### 2.1 The OSINT Process

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Planning     │ ──▶ │  Collection   │ ──▶ │  Processing   │
│   & Direction  │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
                                                     │
┌───────────────┐     ┌───────────────┐              ▼
│ Dissemination  │ ◀── │  Analysis     │ ◀── ┌───────────────┐
│ & Feedback     │     │               │     │  Validation    │
└───────────────┘     └───────────────┘     └───────────────┘
```

### 2.2 OSINT Source Categories

**Technical Sources:**
- DNS records, IP allocations (ARIN, RIPE, APNIC)
- BGP routing tables and AS numbers
- Certificate Transparency logs (crt.sh)
- Internet scanning databases (Shodan, Censys, ZoomEye)
- Code repositories (GitHub, GitLab, Bitbucket)
- Paste sites (Pastebin, GitHub Gists)

**Business Sources:**
- Company websites and press releases
- SEC filings and financial reports
- Patent databases
- Job postings (LinkedIn, Indeed)
- Vendor and partner disclosures

**People Sources:**
- Social media profiles (LinkedIn, Twitter, GitHub)
- Conference presentations and papers
- Email addresses and usernames
- Public records and data breach dumps (Have I Been Pwned)

### 2.3 OSINT Automation

```python
"""
OSINT data collector — aggregates information from multiple
public sources for authorized reconnaissance.

NOTE: This module queries only publicly available data.
Always verify authorization before using against real targets.
"""

import json
import socket
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class OSINTResult:
    """A single piece of intelligence gathered from OSINT."""
    source: str
    category: str
    data: dict
    confidence: float  # 0.0 to 1.0
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class TargetProfile:
    """Aggregated OSINT profile for a target organization."""
    domain: str
    results: list[OSINTResult] = field(default_factory=list)
    subdomains: set[str] = field(default_factory=set)
    ip_addresses: set[str] = field(default_factory=set)
    email_addresses: set[str] = field(default_factory=set)
    technologies: set[str] = field(default_factory=set)
    employees: list[dict] = field(default_factory=list)

    def add_result(self, result: OSINTResult) -> None:
        """Add an OSINT result and update relevant fields."""
        self.results.append(result)

    def summary(self) -> str:
        """Generate a summary of gathered intelligence."""
        lines = [
            f"OSINT Profile: {self.domain}",
            f"{'=' * 50}",
            f"Subdomains discovered:  {len(self.subdomains)}",
            f"IP addresses found:     {len(self.ip_addresses)}",
            f"Email addresses found:  {len(self.email_addresses)}",
            f"Technologies detected:  {len(self.technologies)}",
            f"Employees identified:   {len(self.employees)}",
            f"Total data points:      {len(self.results)}",
        ]
        if self.subdomains:
            lines.append(f"\nSubdomains:")
            for sub in sorted(self.subdomains)[:20]:
                lines.append(f"  - {sub}")
        if self.technologies:
            lines.append(f"\nTechnologies:")
            for tech in sorted(self.technologies):
                lines.append(f"  - {tech}")
        return "\n".join(lines)


def resolve_dns_records(domain: str) -> list[OSINTResult]:
    """
    Resolve common DNS record types for a domain.

    In practice, you would use dnspython for comprehensive queries.
    This simplified version demonstrates the concept.
    """
    results = []
    record_types = ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA"]

    # Simulate DNS resolution (in practice, use dns.resolver)
    try:
        # A record lookup
        ips = socket.getaddrinfo(domain, None, socket.AF_INET)
        unique_ips = set(ip[4][0] for ip in ips)
        for ip in unique_ips:
            results.append(OSINTResult(
                source="DNS",
                category="infrastructure",
                data={"record_type": "A", "domain": domain, "ip": ip},
                confidence=1.0,
            ))
    except socket.gaierror:
        pass

    return results


def check_common_subdomains(domain: str) -> list[str]:
    """
    Check common subdomain prefixes against a domain.

    This is a simplified version — real tools use much larger
    wordlists and concurrent resolution.
    """
    common_prefixes = [
        "www", "mail", "ftp", "smtp", "pop", "imap",
        "admin", "portal", "vpn", "remote", "api",
        "dev", "staging", "test", "beta", "demo",
        "git", "gitlab", "github", "jenkins", "ci",
        "blog", "shop", "store", "app", "mobile",
        "cdn", "static", "media", "assets", "images",
        "ns1", "ns2", "dns", "mx", "relay",
        "db", "database", "mongo", "redis", "elastic",
        "grafana", "prometheus", "kibana", "sentry",
        "jira", "confluence", "wiki", "docs",
        "backup", "old", "legacy", "archive",
    ]

    found = []
    for prefix in common_prefixes:
        subdomain = f"{prefix}.{domain}"
        try:
            socket.getaddrinfo(subdomain, None, socket.AF_INET)
            found.append(subdomain)
        except socket.gaierror:
            pass

    return found


def generate_email_patterns(
    domain: str,
    first_name: str,
    last_name: str,
) -> list[str]:
    """
    Generate common email address patterns for an employee.

    Useful for phishing simulations or credential testing
    in authorized engagements.
    """
    fn = first_name.lower()
    ln = last_name.lower()
    fi = fn[0] if fn else ""
    li = ln[0] if ln else ""

    patterns = [
        f"{fn}.{ln}@{domain}",          # john.doe@company.com
        f"{fn}{ln}@{domain}",            # johndoe@company.com
        f"{fi}{ln}@{domain}",            # jdoe@company.com
        f"{fn}{li}@{domain}",            # johnd@company.com
        f"{fn}_{ln}@{domain}",           # john_doe@company.com
        f"{ln}.{fn}@{domain}",           # doe.john@company.com
        f"{fn}@{domain}",               # john@company.com
        f"{fi}.{ln}@{domain}",           # j.doe@company.com
        f"{fn}{ln[0:3]}@{domain}",       # johndoe@company.com
    ]
    return list(dict.fromkeys(patterns))  # Deduplicate preserving order


# Demonstration
if __name__ == "__main__":
    # Create a target profile (using example.com for safety)
    profile = TargetProfile(domain="example.com")

    # DNS enumeration
    dns_results = resolve_dns_records("example.com")
    for result in dns_results:
        profile.add_result(result)
        profile.ip_addresses.add(result.data.get("ip", ""))

    # Email pattern generation
    patterns = generate_email_patterns("example.com", "Jane", "Smith")
    print("Email patterns generated:")
    for pattern in patterns:
        print(f"  {pattern}")

    print("\n" + profile.summary())
```

---

## 3. DNS Enumeration

DNS is one of the richest sources of information during reconnaissance. Every organization's DNS records reveal infrastructure details.

### 3.1 DNS Record Types for Recon

| Record Type | Information Revealed | Recon Value |
|------------|---------------------|-------------|
| A / AAAA | IPv4/IPv6 addresses | Map infrastructure |
| MX | Mail servers | Identify email provider |
| NS | Name servers | DNS infrastructure |
| TXT | SPF, DKIM, DMARC, verification | Email security posture |
| CNAME | Aliases and CDN usage | Service providers |
| SOA | Primary NS, admin email | Administrative info |
| SRV | Service locations | Internal services |
| PTR | Reverse DNS | Hostname discovery |

### 3.2 DNS Enumeration Commands

```bash
# Basic DNS queries with dig
dig example.com A +short
dig example.com MX +short
dig example.com TXT +short
dig example.com NS +short
dig example.com ANY +noall +answer

# DNS zone transfer attempt (usually blocked)
dig axfr example.com @ns1.example.com

# Reverse DNS lookup
dig -x 93.184.216.34 +short

# DNS enumeration with all record types
for type in A AAAA MX NS TXT CNAME SOA SRV; do
    echo "--- $type ---"
    dig example.com $type +short
done
```

### 3.3 DNS Zone Transfer

A DNS zone transfer (AXFR) copies the entire DNS zone from a name server. If misconfigured, this reveals all hostnames in the zone:

```python
"""
DNS zone transfer checker.

Tests whether a domain's name servers allow zone transfers.
Misconfigured DNS servers may expose the entire zone file,
revealing internal hostnames and IP addresses.

Only test against authorized targets.
"""

import socket
import struct
from typing import Optional


def build_axfr_query(domain: str) -> bytes:
    """
    Build a raw DNS AXFR (zone transfer) query packet.

    This demonstrates the DNS protocol at the binary level.
    In practice, use dnspython: dns.zone.from_xfr()
    """
    # Transaction ID (random)
    import random
    txn_id = random.randint(0, 65535)

    # DNS header
    flags = 0x0000  # Standard query
    qdcount = 1     # One question
    ancount = 0
    nscount = 0
    arcount = 0

    header = struct.pack(
        "!HHHHHH",
        txn_id, flags, qdcount, ancount, nscount, arcount
    )

    # Question section
    question = b""
    for label in domain.split("."):
        question += struct.pack("!B", len(label)) + label.encode()
    question += b"\x00"  # Root label

    # QTYPE=252 (AXFR), QCLASS=1 (IN)
    question += struct.pack("!HH", 252, 1)

    return header + question


def attempt_zone_transfer(
    domain: str,
    nameserver: str,
    timeout: float = 10.0,
) -> Optional[str]:
    """
    Attempt a DNS zone transfer against a nameserver.

    Returns zone data if successful, None if blocked.
    Most properly configured servers will refuse this.
    """
    try:
        # Build query
        query = build_axfr_query(domain)

        # DNS zone transfers use TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((nameserver, 53))

        # TCP DNS requires 2-byte length prefix
        tcp_query = struct.pack("!H", len(query)) + query
        sock.send(tcp_query)

        # Read response length
        length_data = sock.recv(2)
        if len(length_data) < 2:
            return None

        response_length = struct.unpack("!H", length_data)[0]
        response = b""
        while len(response) < response_length:
            chunk = sock.recv(response_length - len(response))
            if not chunk:
                break
            response += chunk

        sock.close()

        # Check RCODE (bits 12-15 of flags)
        if len(response) < 4:
            return None

        flags = struct.unpack("!H", response[2:4])[0]
        rcode = flags & 0x000F

        if rcode == 0:
            return f"Zone transfer SUCCEEDED for {domain} from {nameserver}"
        elif rcode == 5:
            return None  # REFUSED — properly configured
        else:
            return None

    except (socket.timeout, ConnectionRefusedError, OSError):
        return None


def check_zone_transfer_all_ns(domain: str) -> dict:
    """
    Check all nameservers for a domain for zone transfer.

    Returns results for each nameserver.
    """
    results = {}

    # Get nameservers (simplified — use dnspython in practice)
    try:
        ns_records = socket.getaddrinfo(
            f"ns1.{domain}", None, socket.AF_INET
        )
        # This is simplified; real implementation queries NS records
    except socket.gaierror:
        pass

    # For demonstration, we show the structure
    print(f"Checking zone transfer for: {domain}")
    print("In a real engagement, this would query each NS record")
    print("and attempt AXFR against each one.")

    return results


if __name__ == "__main__":
    # Safe demonstration against a known test domain
    print("DNS Zone Transfer Checker")
    print("=" * 40)
    print("\nUsage in authorized testing:")
    print("  1. Enumerate NS records: dig example.com NS")
    print("  2. Attempt AXFR: dig axfr example.com @ns1.example.com")
    print("  3. Or use this script with proper targets")
    print("\nNOTE: Only test domains you are authorized to test.")

    # Show the raw query structure
    query = build_axfr_query("example.com")
    print(f"\nRaw AXFR query ({len(query)} bytes):")
    print(f"  Header:   {query[:12].hex()}")
    print(f"  Question: {query[12:].hex()}")
```

---

## 4. Subdomain Discovery

Discovering subdomains reveals the full scope of an organization's web presence, often uncovering forgotten development servers, staging environments, and administrative panels.

### 4.1 Discovery Methods

**Passive methods** (no direct interaction):
- Certificate Transparency logs (crt.sh)
- Search engine indexing
- DNS aggregation databases (VirusTotal, SecurityTrails)
- Web archive (archive.org)

**Active methods** (direct interaction):
- DNS brute-forcing with wordlists
- Virtual host enumeration
- DNS zone transfer (if misconfigured)
- Web crawling and link extraction

### 4.2 Certificate Transparency

Certificate Transparency (CT) logs are a goldmine for subdomain discovery. Every SSL certificate issued by a trusted CA must be logged, and these logs are publicly searchable.

```python
"""
Subdomain discovery through multiple techniques.

Combines passive and active methods to build a comprehensive
list of subdomains for a target domain.
"""

import json
import socket
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


@dataclass
class SubdomainResult:
    """A discovered subdomain with metadata."""
    subdomain: str
    source: str
    ip_address: Optional[str] = None
    is_alive: bool = False
    http_status: Optional[int] = None
    http_title: Optional[str] = None


class SubdomainEnumerator:
    """
    Multi-source subdomain enumerator.

    Combines passive sources (CT logs) with active resolution
    for comprehensive subdomain discovery.
    """

    # Common subdomain wordlist (abbreviated — real lists have 10,000+ entries)
    COMMON_SUBDOMAINS = [
        "www", "mail", "ftp", "localhost", "webmail", "smtp", "pop",
        "ns1", "ns2", "dns", "dns1", "dns2", "mx", "mx1",
        "admin", "administrator", "portal", "webadmin",
        "api", "api2", "api3", "rest", "graphql",
        "dev", "develop", "development", "staging", "stage",
        "test", "testing", "qa", "uat", "sandbox",
        "beta", "alpha", "demo", "preview",
        "git", "gitlab", "github", "bitbucket", "svn",
        "jenkins", "ci", "cd", "build", "deploy",
        "app", "application", "mobile", "m",
        "blog", "news", "forum", "community", "support",
        "shop", "store", "ecommerce", "cart", "pay",
        "vpn", "remote", "gateway", "proxy",
        "db", "database", "sql", "mysql", "postgres", "mongo",
        "redis", "memcached", "elasticsearch", "elastic",
        "grafana", "prometheus", "kibana", "sentry", "monitoring",
        "cdn", "static", "assets", "media", "images", "img",
        "backup", "bak", "old", "legacy", "archive",
        "internal", "intranet", "extranet", "corp", "corporate",
        "sso", "auth", "login", "oauth", "idp",
        "docs", "documentation", "wiki", "confluence", "jira",
        "status", "health", "healthcheck", "ping",
    ]

    def __init__(self, domain: str, max_workers: int = 10):
        self.domain = domain
        self.max_workers = max_workers
        self.results: dict[str, SubdomainResult] = {}

    def query_crt_sh(self) -> list[str]:
        """
        Query crt.sh Certificate Transparency logs.

        crt.sh provides free access to CT log data.
        """
        url = f"https://crt.sh/?q=%.{self.domain}&output=json"
        subdomains = set()

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            response = urlopen(req, timeout=15)
            data = json.loads(response.read())

            for entry in data:
                name = entry.get("name_value", "")
                # CT entries can contain wildcards and multiple names
                for sub in name.split("\n"):
                    sub = sub.strip().lower()
                    if sub.startswith("*."):
                        sub = sub[2:]
                    if sub.endswith(f".{self.domain}") or sub == self.domain:
                        subdomains.add(sub)

        except (URLError, json.JSONDecodeError, Exception) as e:
            print(f"  [!] crt.sh query failed: {e}")

        return list(subdomains)

    def brute_force(self, wordlist: Optional[list[str]] = None) -> list[str]:
        """
        DNS brute-force with a wordlist.

        Resolves each potential subdomain to check existence.
        """
        if wordlist is None:
            wordlist = self.COMMON_SUBDOMAINS

        found = []

        def check_subdomain(prefix: str) -> Optional[str]:
            subdomain = f"{prefix}.{self.domain}"
            try:
                socket.getaddrinfo(subdomain, None, socket.AF_INET)
                return subdomain
            except socket.gaierror:
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(check_subdomain, prefix): prefix
                for prefix in wordlist
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    found.append(result)

        return found

    def resolve_subdomain(self, subdomain: str) -> SubdomainResult:
        """Resolve a subdomain and gather additional information."""
        result = SubdomainResult(
            subdomain=subdomain,
            source="resolution",
        )

        try:
            ips = socket.getaddrinfo(subdomain, None, socket.AF_INET)
            if ips:
                result.ip_address = ips[0][4][0]
                result.is_alive = True
        except socket.gaierror:
            pass

        return result

    def enumerate(self) -> list[SubdomainResult]:
        """
        Run full subdomain enumeration pipeline.

        Combines passive and active techniques.
        """
        all_subdomains = set()

        # Passive: Certificate Transparency
        print(f"[*] Querying Certificate Transparency for {self.domain}...")
        ct_subs = self.query_crt_sh()
        print(f"  [+] Found {len(ct_subs)} subdomains via CT logs")
        all_subdomains.update(ct_subs)

        # Active: DNS brute-force
        print(f"[*] Brute-forcing common subdomains...")
        bf_subs = self.brute_force()
        print(f"  [+] Found {len(bf_subs)} subdomains via brute-force")
        all_subdomains.update(bf_subs)

        # Resolve all found subdomains
        print(f"[*] Resolving {len(all_subdomains)} unique subdomains...")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.resolve_subdomain, sub): sub
                for sub in all_subdomains
            }
            for future in as_completed(futures):
                result = future.result()
                if result.is_alive:
                    results.append(result)

        self.results = {r.subdomain: r for r in results}
        return results

    def report(self) -> str:
        """Generate a subdomain enumeration report."""
        lines = [
            f"Subdomain Enumeration Report: {self.domain}",
            "=" * 60,
            f"Total unique subdomains resolved: {len(self.results)}",
            "",
        ]
        for sub, result in sorted(self.results.items()):
            ip = result.ip_address or "N/A"
            lines.append(f"  {sub:40s} -> {ip}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Safe demonstration
    print("Subdomain Enumerator")
    print("=" * 40)
    print("Usage: enumerator = SubdomainEnumerator('target.com')")
    print("       results = enumerator.enumerate()")
    print("       print(enumerator.report())")
    print("\nOnly use against authorized targets.")

    # Show wordlist size
    print(f"\nBuilt-in wordlist: {len(SubdomainEnumerator.COMMON_SUBDOMAINS)} entries")
    print("First 10 entries:", SubdomainEnumerator.COMMON_SUBDOMAINS[:10])
```

---

## 5. Google Dorking

Google dorking (Google hacking) uses advanced search operators to find information that organizations have inadvertently exposed to search engine crawlers.

### 5.1 Essential Search Operators

| Operator | Purpose | Example |
|----------|---------|---------|
| `site:` | Limit to specific domain | `site:example.com` |
| `inurl:` | Search in URL path | `inurl:admin` |
| `intitle:` | Search page titles | `intitle:"index of"` |
| `filetype:` | Find specific file types | `filetype:pdf` |
| `intext:` | Search page content | `intext:"password"` |
| `ext:` | File extension | `ext:sql` |
| `cache:` | Cached version of page | `cache:example.com` |
| `-` | Exclude term | `site:example.com -www` |
| `""` | Exact match | `"error in your SQL syntax"` |
| `*` | Wildcard | `site:*.example.com` |

### 5.2 Common Google Dorks

```python
"""
Google dork generator for authorized reconnaissance.

Generates search queries to find potentially exposed
information. Always verify authorization before accessing
any discovered resources.
"""

from dataclasses import dataclass
from enum import Enum


class DorkCategory(Enum):
    SENSITIVE_FILES = "Sensitive Files"
    LOGIN_PAGES = "Login Pages"
    ERROR_MESSAGES = "Error Messages"
    DIRECTORY_LISTINGS = "Directory Listings"
    DATABASE_DUMPS = "Database Dumps"
    CONFIG_FILES = "Configuration Files"
    BACKUP_FILES = "Backup Files"
    SOURCE_CODE = "Source Code"
    CREDENTIALS = "Credentials"
    INFRASTRUCTURE = "Infrastructure"


@dataclass
class GoogleDork:
    """A Google dork query with metadata."""
    query_template: str
    category: DorkCategory
    description: str
    risk_level: str  # info, low, medium, high, critical

    def for_domain(self, domain: str) -> str:
        """Generate the dork query for a specific domain."""
        return self.query_template.replace("{domain}", domain)


# Standard Google dork database
DORK_DATABASE = [
    # Sensitive Files
    GoogleDork(
        'site:{domain} filetype:pdf "confidential"',
        DorkCategory.SENSITIVE_FILES,
        "Find PDF files marked as confidential",
        "medium",
    ),
    GoogleDork(
        'site:{domain} filetype:xlsx OR filetype:csv "password" OR "username"',
        DorkCategory.SENSITIVE_FILES,
        "Find spreadsheets containing credential data",
        "high",
    ),
    GoogleDork(
        'site:{domain} filetype:doc OR filetype:docx "internal" OR "draft"',
        DorkCategory.SENSITIVE_FILES,
        "Find internal or draft documents",
        "low",
    ),

    # Login Pages
    GoogleDork(
        'site:{domain} inurl:login OR inurl:signin OR inurl:admin',
        DorkCategory.LOGIN_PAGES,
        "Find authentication pages",
        "info",
    ),
    GoogleDork(
        'site:{domain} intitle:"admin panel" OR intitle:"dashboard"',
        DorkCategory.LOGIN_PAGES,
        "Find admin panels and dashboards",
        "medium",
    ),

    # Error Messages
    GoogleDork(
        'site:{domain} "error in your SQL syntax" OR "mysql_fetch"',
        DorkCategory.ERROR_MESSAGES,
        "Find SQL error messages (potential SQLi)",
        "high",
    ),
    GoogleDork(
        'site:{domain} "Fatal error" OR "Warning:" filetype:php',
        DorkCategory.ERROR_MESSAGES,
        "Find PHP error pages with stack traces",
        "medium",
    ),
    GoogleDork(
        'site:{domain} "Traceback (most recent call last)"',
        DorkCategory.ERROR_MESSAGES,
        "Find Python stack traces",
        "medium",
    ),

    # Directory Listings
    GoogleDork(
        'site:{domain} intitle:"index of" "parent directory"',
        DorkCategory.DIRECTORY_LISTINGS,
        "Find open directory listings",
        "medium",
    ),
    GoogleDork(
        'site:{domain} intitle:"index of" ".git"',
        DorkCategory.DIRECTORY_LISTINGS,
        "Find exposed .git directories",
        "critical",
    ),

    # Configuration Files
    GoogleDork(
        'site:{domain} filetype:env OR filetype:ini OR filetype:cfg',
        DorkCategory.CONFIG_FILES,
        "Find configuration files",
        "high",
    ),
    GoogleDork(
        'site:{domain} filetype:xml "password" OR "secret"',
        DorkCategory.CONFIG_FILES,
        "Find XML configs with credentials",
        "high",
    ),
    GoogleDork(
        'site:{domain} filetype:yaml OR filetype:yml "apikey" OR "api_key"',
        DorkCategory.CONFIG_FILES,
        "Find YAML files with API keys",
        "critical",
    ),

    # Backup Files
    GoogleDork(
        'site:{domain} filetype:bak OR filetype:old OR filetype:backup',
        DorkCategory.BACKUP_FILES,
        "Find backup files",
        "medium",
    ),
    GoogleDork(
        'site:{domain} filetype:sql "INSERT INTO" OR "CREATE TABLE"',
        DorkCategory.DATABASE_DUMPS,
        "Find SQL database dumps",
        "critical",
    ),

    # Source Code
    GoogleDork(
        'site:github.com "{domain}" password OR secret OR api_key',
        DorkCategory.SOURCE_CODE,
        "Find leaked credentials in GitHub repos",
        "critical",
    ),
    GoogleDork(
        'site:pastebin.com "{domain}"',
        DorkCategory.SOURCE_CODE,
        "Find paste site mentions of the target",
        "medium",
    ),
]


def generate_dork_report(domain: str) -> str:
    """Generate a categorized dork report for a domain."""
    lines = [
        f"Google Dork Report for: {domain}",
        "=" * 60,
        "",
        "NOTE: Only access discovered resources if you have authorization.",
        "",
    ]

    categories = sorted(set(d.category for d in DORK_DATABASE),
                       key=lambda c: c.value)
    for cat in categories:
        lines.append(f"\n--- {cat.value} ---")
        dorks = [d for d in DORK_DATABASE if d.category == cat]
        for dork in dorks:
            query = dork.for_domain(domain)
            lines.append(f"  [{dork.risk_level.upper():8s}] {dork.description}")
            lines.append(f"           Query: {query}")
    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_dork_report("example.com")
    print(report)
```

---

## 6. Shodan and Internet-Wide Scanning

Shodan, Censys, and ZoomEye continuously scan the internet and index information about connected devices. These databases allow passive reconnaissance of internet-facing assets.

### 6.1 Shodan Search Filters

| Filter | Purpose | Example |
|--------|---------|---------|
| `hostname:` | Search by hostname | `hostname:example.com` |
| `ip:` | Search by IP address | `ip:93.184.216.34` |
| `port:` | Search by open port | `port:3389` |
| `org:` | Search by organization | `org:"Example Corp"` |
| `product:` | Search by software | `product:Apache` |
| `version:` | Search by version | `version:2.4.49` |
| `ssl:` | Search SSL cert fields | `ssl:"example.com"` |
| `vuln:` | Search by CVE | `vuln:CVE-2021-44228` |
| `country:` | Filter by country | `country:US` |
| `city:` | Filter by city | `city:"San Francisco"` |

### 6.2 Useful Shodan Queries

```
# Find exposed databases
product:MongoDB port:27017 -authentication
product:Elasticsearch port:9200

# Find vulnerable web servers
http.title:"Index of /" port:80
Apache/2.4.49 country:US

# Find IoT devices
"Server: Hikvision" port:80
"Server: DVRDVS" port:80

# Find industrial control systems
port:502 Modbus
port:47808 "BACnet"

# Find organization's assets
org:"Target Corp" port:443
ssl:"targetcorp.com"
```

---

## 7. WHOIS and Domain Intelligence

WHOIS records provide domain registration information, though modern privacy protections (GDPR-compliant redaction) limit available data.

```python
"""
WHOIS and domain intelligence gathering.

Extracts registration information, name servers, and
historical data from domain records.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class WHOISRecord:
    """Parsed WHOIS record data."""
    domain: str
    registrar: str = ""
    creation_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    name_servers: list[str] = field(default_factory=list)
    status: list[str] = field(default_factory=list)
    registrant_org: str = ""
    registrant_country: str = ""
    dnssec: str = ""
    privacy_protected: bool = False

    @property
    def domain_age_days(self) -> Optional[int]:
        if self.creation_date:
            return (datetime.now() - self.creation_date).days
        return None

    @property
    def days_until_expiry(self) -> Optional[int]:
        if self.expiration_date:
            return (self.expiration_date - datetime.now()).days
        return None

    def security_observations(self) -> list[str]:
        """Generate security-relevant observations from WHOIS data."""
        observations = []

        if self.privacy_protected:
            observations.append(
                "WHOIS privacy is enabled — registrant details are redacted"
            )

        if self.days_until_expiry and self.days_until_expiry < 30:
            observations.append(
                f"Domain expires in {self.days_until_expiry} days — "
                "potential for domain takeover if not renewed"
            )

        if self.dnssec.lower() in ("unsigned", ""):
            observations.append(
                "DNSSEC is not enabled — vulnerable to DNS spoofing"
            )

        if len(self.name_servers) < 2:
            observations.append(
                "Single name server — no DNS redundancy"
            )

        return observations


def analyze_domain(domain: str) -> str:
    """
    Comprehensive domain analysis combining WHOIS
    and DNS information.
    """
    # In practice, use python-whois library
    # This demonstrates the analysis structure
    record = WHOISRecord(
        domain=domain,
        registrar="Example Registrar, Inc.",
        creation_date=datetime(2010, 3, 15),
        expiration_date=datetime(2026, 3, 15),
        name_servers=["ns1.example.com", "ns2.example.com"],
        status=["clientTransferProhibited"],
        registrant_org="Example Organization",
        registrant_country="US",
        dnssec="unsigned",
        privacy_protected=True,
    )

    lines = [
        f"Domain Analysis: {domain}",
        "=" * 50,
        f"Registrar:     {record.registrar}",
        f"Created:       {record.creation_date}",
        f"Expires:       {record.expiration_date}",
        f"Domain Age:    {record.domain_age_days} days",
        f"Days to Expiry: {record.days_until_expiry}",
        f"Name Servers:  {', '.join(record.name_servers)}",
        f"DNSSEC:        {record.dnssec}",
        f"Privacy:       {'Yes' if record.privacy_protected else 'No'}",
        "",
        "Security Observations:",
    ]
    for obs in record.security_observations():
        lines.append(f"  [!] {obs}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(analyze_domain("example.com"))
```

---

## 8. Metadata Extraction

Documents published on websites often contain metadata revealing internal information: author names, software versions, file paths, printer names, and even GPS coordinates in images.

### 8.1 Common Metadata Sources

| File Type | Metadata Available |
|-----------|-------------------|
| PDF | Author, creation tool, modification dates, embedded fonts |
| DOCX/XLSX | Author, company, revision count, template paths |
| JPEG/PNG | EXIF data: camera model, GPS coordinates, timestamps |
| EXE/DLL | Compiler version, debug paths, digital signatures |

### 8.2 Metadata Extraction Tool

```python
"""
Document metadata extractor for authorized reconnaissance.

Extracts hidden metadata from documents that may reveal
internal information about the target organization.
"""

import struct
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional


@dataclass
class DocumentMetadata:
    """Extracted metadata from a document."""
    filename: str
    file_type: str
    author: str = ""
    creator_tool: str = ""
    creation_date: str = ""
    modification_date: str = ""
    company: str = ""
    title: str = ""
    subject: str = ""
    keywords: list[str] = field(default_factory=list)
    custom_properties: dict = field(default_factory=dict)
    internal_paths: list[str] = field(default_factory=list)

    def security_findings(self) -> list[str]:
        """Identify security-relevant metadata."""
        findings = []
        if self.author:
            findings.append(f"Author name disclosed: {self.author}")
        if self.company:
            findings.append(f"Company name in metadata: {self.company}")
        if self.creator_tool:
            findings.append(f"Software used: {self.creator_tool}")
        if self.internal_paths:
            findings.append(
                f"Internal file paths exposed: {', '.join(self.internal_paths[:5])}"
            )
        return findings


def extract_docx_metadata(filepath: str) -> Optional[DocumentMetadata]:
    """
    Extract metadata from a DOCX file.

    DOCX files are ZIP archives containing XML files.
    Core properties are in docProps/core.xml.
    """
    metadata = DocumentMetadata(
        filename=Path(filepath).name,
        file_type="DOCX",
    )

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            # Core properties
            if "docProps/core.xml" in zf.namelist():
                core = ET.fromstring(zf.read("docProps/core.xml"))
                ns = {
                    "dc": "http://purl.org/dc/elements/1.1/",
                    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
                    "dcterms": "http://purl.org/dc/terms/",
                }
                creator = core.find("dc:creator", ns)
                if creator is not None and creator.text:
                    metadata.author = creator.text

                title = core.find("dc:title", ns)
                if title is not None and title.text:
                    metadata.title = title.text

                modified = core.find("dcterms:modified", ns)
                if modified is not None and modified.text:
                    metadata.modification_date = modified.text

            # App properties (software info)
            if "docProps/app.xml" in zf.namelist():
                app = ET.fromstring(zf.read("docProps/app.xml"))
                ns_app = {
                    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
                }
                company = app.find("ep:Company", ns_app)
                if company is not None and company.text:
                    metadata.company = company.text

                app_name = app.find("ep:Application", ns_app)
                if app_name is not None and app_name.text:
                    metadata.creator_tool = app_name.text

    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return None

    return metadata


def extract_pdf_metadata(filepath: str) -> Optional[DocumentMetadata]:
    """
    Extract metadata from a PDF file (basic extraction).

    For comprehensive PDF metadata, use PyPDF2 or pikepdf.
    This demonstrates manual extraction from the PDF trailer.
    """
    metadata = DocumentMetadata(
        filename=Path(filepath).name,
        file_type="PDF",
    )

    try:
        with open(filepath, "rb") as f:
            content = f.read()

        # Find the /Info dictionary in the PDF
        # This is a simplified parser — real PDFs have complex structure
        text = content.decode("latin-1", errors="ignore")

        # Extract common metadata fields
        import re
        patterns = {
            "author": r"/Author\s*\(([^)]*)\)",
            "creator_tool": r"/Creator\s*\(([^)]*)\)",
            "title": r"/Title\s*\(([^)]*)\)",
            "subject": r"/Subject\s*\(([^)]*)\)",
            "creation_date": r"/CreationDate\s*\(([^)]*)\)",
        }

        for field_name, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                setattr(metadata, field_name, match.group(1))

    except (OSError, UnicodeDecodeError):
        return None

    return metadata


if __name__ == "__main__":
    print("Document Metadata Extractor")
    print("=" * 40)
    print("\nSupported formats: DOCX, PDF")
    print("Usage:")
    print("  metadata = extract_docx_metadata('report.docx')")
    print("  findings = metadata.security_findings()")
    print("\nSecurity-relevant metadata may include:")
    print("  - Author names (employee identification)")
    print("  - Company names (organizational confirmation)")
    print("  - Software versions (attack surface mapping)")
    print("  - Internal file paths (network structure)")
    print("  - GPS coordinates in images (physical location)")
```

---

## 9. Social Media and Employee OSINT

Social media and professional networking sites provide rich intelligence about an organization's employees, technologies, and internal processes.

### 9.1 LinkedIn Intelligence

LinkedIn is particularly valuable for:
- **Employee enumeration**: Building a list of employees and their roles
- **Technology stack**: Job postings reveal the technologies used
- **Organizational structure**: Management hierarchy and team sizes
- **Email patterns**: Combined with domain info to guess email addresses
- **Security posture clues**: Job postings for security roles indicate priorities

### 9.2 GitHub and Code Repository OSINT

Developers frequently leak sensitive information in public repositories:

- API keys and tokens in commit history
- Internal URLs and IP addresses in configuration files
- Database credentials in environment files
- Infrastructure details in CI/CD configurations
- Email addresses and internal usernames in commit logs

---

## 10. Automated Reconnaissance Framework

```python
"""
Automated reconnaissance framework.

Orchestrates multiple recon modules into a single
pipeline with structured output.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional


@dataclass
class ReconModule:
    """A pluggable reconnaissance module."""
    name: str
    description: str
    recon_type: str  # passive, active
    function: Optional[Callable] = None
    enabled: bool = True


@dataclass
class ReconFinding:
    """A single finding from reconnaissance."""
    module: str
    finding_type: str
    value: str
    confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ReconReport:
    """Complete reconnaissance report."""
    target: str
    start_time: str = ""
    end_time: str = ""
    findings: list[ReconFinding] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_finding(self, finding: ReconFinding) -> None:
        self.findings.append(finding)

    def generate_summary(self) -> dict:
        """Generate a summary of all findings."""
        self.summary = {
            "total_findings": len(self.findings),
            "by_module": {},
            "by_type": {},
        }
        for f in self.findings:
            self.summary["by_module"][f.module] = (
                self.summary["by_module"].get(f.module, 0) + 1
            )
            self.summary["by_type"][f.finding_type] = (
                self.summary["by_type"].get(f.finding_type, 0) + 1
            )
        return self.summary

    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps(asdict(self), indent=2)

    def to_text(self) -> str:
        """Export report as human-readable text."""
        lines = [
            f"Reconnaissance Report: {self.target}",
            f"Period: {self.start_time} to {self.end_time}",
            "=" * 60,
            "",
        ]
        self.generate_summary()
        lines.append(f"Total findings: {self.summary['total_findings']}")
        lines.append("\nFindings by module:")
        for mod, count in self.summary["by_module"].items():
            lines.append(f"  {mod}: {count}")
        lines.append("\nFindings by type:")
        for typ, count in self.summary["by_type"].items():
            lines.append(f"  {typ}: {count}")

        lines.append("\n\nDetailed Findings:")
        lines.append("-" * 60)
        for i, f in enumerate(self.findings, 1):
            lines.append(
                f"\n[{i}] [{f.module}] {f.finding_type}: {f.value}"
            )
            if f.metadata:
                for k, v in f.metadata.items():
                    lines.append(f"     {k}: {v}")
        return "\n".join(lines)


class ReconFramework:
    """
    Orchestrates multiple reconnaissance modules.

    Usage:
        framework = ReconFramework("target.com")
        framework.register_module(...)
        report = framework.run()
    """

    def __init__(self, target: str):
        self.target = target
        self.modules: list[ReconModule] = []
        self.report = ReconReport(target=target)

    def register_module(self, module: ReconModule) -> None:
        self.modules.append(module)

    def run(self, passive_only: bool = False) -> ReconReport:
        self.report.start_time = datetime.utcnow().isoformat()

        for module in self.modules:
            if not module.enabled:
                continue
            if passive_only and module.recon_type == "active":
                print(f"  [SKIP] {module.name} (active module, passive-only mode)")
                continue

            print(f"  [RUN]  {module.name}...")
            if module.function:
                try:
                    findings = module.function(self.target)
                    for f in findings:
                        self.report.add_finding(f)
                    print(f"  [OK]   {len(findings)} findings")
                except Exception as e:
                    print(f"  [ERR]  {module.name}: {e}")

        self.report.end_time = datetime.utcnow().isoformat()
        self.report.generate_summary()
        return self.report


# Example module implementations
def dns_recon_module(target: str) -> list[ReconFinding]:
    """Simple DNS reconnaissance module."""
    import socket
    findings = []
    try:
        ips = socket.getaddrinfo(target, None, socket.AF_INET)
        for ip_info in ips:
            findings.append(ReconFinding(
                module="dns_recon",
                finding_type="ip_address",
                value=ip_info[4][0],
                confidence=1.0,
                metadata={"record_type": "A"},
            ))
    except socket.gaierror:
        pass
    return findings


if __name__ == "__main__":
    framework = ReconFramework("example.com")
    framework.register_module(ReconModule(
        name="DNS Reconnaissance",
        description="Resolve DNS records for the target",
        recon_type="passive",
        function=dns_recon_module,
    ))

    print("Starting reconnaissance...\n")
    report = framework.run()
    print("\n" + report.to_text())
```

---

## 11. Countermeasures and Detection

Understanding how to detect and prevent reconnaissance helps both attackers (to be stealthier) and defenders (to catch recon early).

### 11.1 Defensive Measures

| Recon Technique | Countermeasure |
|----------------|----------------|
| DNS enumeration | Minimize public DNS records, use split-horizon DNS |
| Subdomain discovery | Use wildcard DNS to hide subdomains, CT monitoring |
| Google dorking | robots.txt, remove indexed sensitive content |
| Shodan/Censys | Minimize exposed services, use firewalls |
| Metadata leakage | Strip metadata from published documents |
| Social media OSINT | Employee security awareness training |
| GitHub leaks | Pre-commit hooks, secret scanning |

### 11.2 Detection Indicators

- High volume of DNS queries from a single source
- Sequential port scanning patterns in firewall logs
- Unusual web crawling patterns in access logs
- Multiple failed authentication attempts (credential testing)
- Directory enumeration patterns (sequential 404 errors)

---

## 12. Exercises

1. **Passive Recon**: Using only passive techniques (no direct target interaction), gather as much information as possible about a target domain from a CTF practice platform.

2. **DNS Deep Dive**: Write a Python script using dnspython that queries all record types for a domain and generates a formatted report.

3. **Subdomain Race**: Use three different subdomain enumeration tools (subfinder, amass, crt.sh) against a practice target. Compare coverage and speed.

4. **Google Dork Audit**: Run the Google dork generator against a domain you control. Document any findings and remediate exposed information.

5. **Metadata Audit**: Download 10 public PDFs from a government website. Extract metadata and analyze what internal information is revealed.

6. **Recon Automation**: Extend the ReconFramework to include at least 5 modules covering different reconnaissance techniques. Generate a comprehensive report.

---

## 13. Summary

Reconnaissance is the foundation of effective penetration testing:

- **Passive recon** gathers intelligence without touching the target — OSINT, CT logs, search engines
- **Active recon** directly interacts with target systems — DNS queries, port scans, web crawling
- **DNS enumeration** reveals infrastructure: subdomains, mail servers, cloud providers
- **Google dorking** finds accidentally exposed files, credentials, and error messages
- **Shodan/Censys** provide pre-scanned data about internet-facing assets
- **Metadata** in documents leaks internal information about organizations
- **Automation** enables consistent, comprehensive reconnaissance across engagements

The quality of reconnaissance directly determines the success of exploitation phases that follow.

---

## 14. References

- OSINT Framework: https://osintframework.com/
- Shodan: https://www.shodan.io/
- crt.sh (Certificate Transparency): https://crt.sh/
- Google Hacking Database (GHDB): https://www.exploit-db.com/google-hacking-database
- theHarvester: https://github.com/laramies/theHarvester
- Amass: https://github.com/owasp-amass/amass
- Subfinder: https://github.com/projectdiscovery/subfinder
- Recon-ng: https://github.com/lanmaster53/recon-ng
