"""
Exercise Solutions: Incident Response and Forensics
====================================================
Lesson 14 from Security topic.

Covers log analysis, IOC databases, incident response playbooks,
memory forensics guides, post-incident reports, and network analysis.
"""

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Exercise 1: Log Analysis
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    ip: str
    user: str
    timestamp: str
    method: str
    path: str
    status: int
    size: int


@dataclass
class SecurityIncident:
    classification: str
    severity: str
    source_ip: str
    summary: str
    evidence: list[str]


def parse_log_line(line: str) -> Optional[LogEntry]:
    """Parse an Apache/Nginx-style log line."""
    pattern = (
        r'(\d+\.\d+\.\d+\.\d+)\s+-\s+(\S+)\s+'
        r'\[([^\]]+)\]\s+"(\w+)\s+([^\s]+)\s+[^"]+"\s+(\d+)\s+(\d+)'
    )
    match = re.match(pattern, line)
    if match:
        return LogEntry(
            ip=match.group(1),
            user=match.group(2) if match.group(2) != "-" else "",
            timestamp=match.group(3),
            method=match.group(4),
            path=match.group(5),
            status=int(match.group(6)),
            size=int(match.group(7)),
        )
    return None


def analyze_logs(log_lines: list[str]) -> list[SecurityIncident]:
    """Analyze log lines and identify security incidents."""
    entries = [parse_log_line(line) for line in log_lines]
    entries = [e for e in entries if e is not None]

    incidents = []

    # Group by IP
    by_ip: dict[str, list[LogEntry]] = defaultdict(list)
    for entry in entries:
        by_ip[entry.ip].append(entry)

    for ip, ip_entries in by_ip.items():
        # Detect brute force (multiple 401s in short time)
        auth_failures = [e for e in ip_entries if e.status == 401]
        if len(auth_failures) >= 3:
            incidents.append(SecurityIncident(
                classification="Brute Force Attack",
                severity="MEDIUM",
                source_ip=ip,
                summary=f"{len(auth_failures)} failed login attempts from {ip}",
                evidence=[f"  {e.timestamp}: {e.method} {e.path} -> {e.status}"
                          for e in auth_failures],
            ))

        # Detect path traversal
        traversal = [e for e in ip_entries if ".." in e.path or "/etc/" in e.path]
        if traversal:
            incidents.append(SecurityIncident(
                classification="Path Traversal Attempt",
                severity="HIGH",
                source_ip=ip,
                summary=f"Directory traversal attempt from {ip}",
                evidence=[f"  {e.path}" for e in traversal],
            ))

        # Detect SQL injection attempts
        sqli = [e for e in ip_entries
                if any(p in e.path.lower() for p in ["' or", "1=1", "--", "union"])]
        if sqli:
            incidents.append(SecurityIncident(
                classification="SQL Injection Attempt",
                severity="CRITICAL",
                source_ip=ip,
                summary=f"SQL injection attempt from {ip}",
                evidence=[f"  {e.path}" for e in sqli],
            ))

        # Detect XSS attempts
        xss = [e for e in ip_entries
               if any(p in e.path.lower() for p in ["<script>", "alert(", "onerror="])]
        if xss:
            incidents.append(SecurityIncident(
                classification="XSS Attempt",
                severity="MEDIUM",
                source_ip=ip,
                summary=f"Cross-site scripting attempt from {ip}",
                evidence=[f"  {e.path}" for e in xss],
            ))

        # Detect data exfiltration (large responses from admin endpoints)
        admin_exports = [e for e in ip_entries
                         if "/admin/export" in e.path and e.size > 100000]
        if admin_exports:
            total_size = sum(e.size for e in admin_exports)
            incidents.append(SecurityIncident(
                classification="Potential Data Exfiltration",
                severity="HIGH",
                source_ip=ip,
                summary=f"Large data exports from admin panel ({total_size:,} bytes)",
                evidence=[f"  {e.path} -> {e.size:,} bytes" for e in admin_exports],
            ))

    return incidents


def exercise_1_log_analysis():
    """Analyze sample log entries for security incidents."""
    log_lines = [
        '192.168.1.50 - - [15/Jan/2025:10:00:01 +0000] "GET /login HTTP/1.1" 200 1234',
        '192.168.1.50 - - [15/Jan/2025:10:00:02 +0000] "POST /login HTTP/1.1" 401 89',
        '192.168.1.50 - - [15/Jan/2025:10:00:03 +0000] "POST /login HTTP/1.1" 401 89',
        '192.168.1.50 - - [15/Jan/2025:10:00:04 +0000] "POST /login HTTP/1.1" 401 89',
        '10.0.0.5 - admin [15/Jan/2025:10:05:00 +0000] "GET /admin/users HTTP/1.1" 200 5678',
        '10.0.0.5 - admin [15/Jan/2025:10:05:01 +0000] "GET /admin/export?table=users HTTP/1.1" 200 890123',
        '10.0.0.5 - admin [15/Jan/2025:10:05:02 +0000] "GET /admin/export?table=payments HTTP/1.1" 200 1234567',
        '203.0.113.10 - - [15/Jan/2025:10:10:00 +0000] "GET /../../etc/passwd HTTP/1.1" 403 0',
        "203.0.113.10 - - [15/Jan/2025:10:10:01 +0000] \"GET /search?q=' OR 1=1 -- HTTP/1.1\" 500 0",
        '203.0.113.10 - - [15/Jan/2025:10:10:02 +0000] "GET /search?q=<script>alert(1)</script> HTTP/1.1" 200 456',
    ]

    incidents = analyze_logs(log_lines)

    print("Security Incident Analysis")
    print("=" * 60)
    for inc in incidents:
        print(f"\n  [{inc.severity}] {inc.classification}")
        print(f"  Source: {inc.source_ip}")
        print(f"  Summary: {inc.summary}")
        print(f"  Evidence:")
        for ev in inc.evidence:
            print(f"    {ev}")


# ---------------------------------------------------------------------------
# Exercise 2: IOC Database
# ---------------------------------------------------------------------------

def exercise_2_ioc_database():
    """Create and demonstrate an IOC (Indicator of Compromise) database."""
    ioc_database = {
        "malicious_ips": [
            {"value": "198.51.100.10", "type": "C2 Server", "confidence": "high"},
            {"value": "203.0.113.50", "type": "Scanning", "confidence": "medium"},
            {"value": "192.0.2.100", "type": "Phishing", "confidence": "high"},
            {"value": "198.51.100.200", "type": "Malware Distribution", "confidence": "high"},
            {"value": "203.0.113.75", "type": "Brute Force", "confidence": "medium"},
        ],
        "malicious_domains": [
            {"value": "evil-payload.example.com", "type": "Malware Distribution"},
            {"value": "phishing-login.example.com", "type": "Phishing"},
            {"value": "c2-server.example.com", "type": "Command and Control"},
            {"value": "data-exfil.example.com", "type": "Data Exfiltration"},
            {"value": "crypto-miner.example.com", "type": "Cryptomining"},
        ],
        "malware_hashes": [
            {"value": "d41d8cd98f00b204e9800998ecf8427e", "algo": "md5",
             "malware": "TrojanDropper"},
            {"value": "e3b0c44298fc1c149afbf4c8996fb924", "algo": "sha256",
             "malware": "Ransomware.Maze"},
            {"value": "a94a8fe5ccb19ba61c4c0873d391e987", "algo": "sha1",
             "malware": "Backdoor.Agent"},
            {"value": "098f6bcd4621d373cade4e832627b4f6", "algo": "md5",
             "malware": "Keylogger.Generic"},
            {"value": "5eb63bbbe01eeed093cb22bb8f5acdc3", "algo": "md5",
             "malware": "InfoStealer"},
        ],
        "suspicious_filenames": [
            "svchost.exe.bak",
            "csrss32.exe",
            "update.ps1",
            "payload.dll",
            "mimikatz.exe",
        ],
    }

    print("IOC Database")
    print("=" * 60)
    for category, items in ioc_database.items():
        print(f"\n  {category}: {len(items)} entries")
        for item in items[:3]:
            print(f"    {item}")

    # Demonstrate IOC scanning
    test_files = [
        "normal_document.pdf",
        "svchost.exe.bak",  # Suspicious
        "report.xlsx",
        "update.ps1",        # Suspicious
    ]

    suspicious_names = set(ioc_database["suspicious_filenames"])
    print("\nIOC Scan Results:")
    for f in test_files:
        if f in suspicious_names:
            print(f"  [ALERT] {f} matches IOC database!")
        else:
            print(f"  [OK] {f}")

    return ioc_database


# ---------------------------------------------------------------------------
# Exercise 3: Incident Response Playbook (SQL Injection)
# ---------------------------------------------------------------------------

def exercise_3_ir_playbook():
    """Generate an SQL injection incident response playbook."""
    playbook = {
        "title": "SQL Injection Attack Response Playbook",
        "trigger_conditions": [
            "WAF alerts for SQL injection patterns (UNION, OR 1=1, etc.)",
            "Application errors containing SQL syntax messages",
            "SIEM correlation rule: >5 SQL error responses in 1 minute",
            "Database audit log: unusual query patterns or data access",
        ],
        "phases": {
            "1. Preparation": {
                "actions": [
                    "Maintain WAF with updated SQL injection rules",
                    "Enable database audit logging (queries, logins, schema changes)",
                    "Set up application error alerting",
                    "Train developers on parameterized queries",
                    "Maintain a database schema inventory",
                ],
                "tools": ["WAF (ModSecurity/AWS WAF)", "Database audit logs",
                          "SIEM (Splunk/ELK)", "Application error tracking"],
            },
            "2. Detection & Analysis": {
                "actions": [
                    "Confirm the attack: review WAF logs for injection patterns",
                    "Identify attack vector: which endpoint(s) are targeted",
                    "Determine scope: was data actually extracted?",
                    "Check database audit logs for unusual queries",
                    "Identify the attacker's IP and user agent",
                ],
                "commands": [
                    "grep -i 'union\\|select.*from\\|or 1=1' /var/log/nginx/access.log",
                    "SELECT * FROM pg_stat_activity WHERE state = 'active';",
                ],
            },
            "3. Containment": {
                "actions": [
                    "Block attacker IP at WAF/firewall level",
                    "If exploitation confirmed: take affected endpoint offline",
                    "Revoke compromised database credentials",
                    "Enable enhanced logging for forensic collection",
                ],
                "commands": [
                    "iptables -A INPUT -s <attacker_ip> -j DROP",
                    "ALTER USER app_user WITH PASSWORD 'new_password';",
                ],
            },
            "4. Eradication & Recovery": {
                "actions": [
                    "Fix the vulnerable code (parameterized queries)",
                    "Scan codebase for similar vulnerabilities (Semgrep/Bandit)",
                    "Deploy patched code through CI/CD pipeline",
                    "Verify fix with SQL injection testing (sqlmap in auth mode)",
                    "Monitor for continued attack attempts",
                ],
            },
            "5. Lessons Learned": {
                "actions": [
                    "Conduct post-incident review within 48 hours",
                    "Update SAST rules to catch this pattern",
                    "Add SQL injection tests to CI/CD pipeline",
                    "Review and update this playbook",
                    "Determine if breach notification is required (data accessed)",
                ],
            },
        },
        "escalation": {
            "Severity 1 (Data confirmed exfiltrated)": "CISO + Legal + DPO within 1 hour",
            "Severity 2 (Attack detected, no confirmed breach)": "Security team lead within 2 hours",
            "Severity 3 (Blocked by WAF, no impact)": "Security team during business hours",
        },
    }

    print("SQL Injection Incident Response Playbook")
    print("=" * 60)
    print(f"\nTrigger Conditions:")
    for tc in playbook["trigger_conditions"]:
        print(f"  - {tc}")

    for phase, content in playbook["phases"].items():
        print(f"\n{phase}:")
        for action in content["actions"]:
            print(f"  - {action}")
        if "commands" in content:
            print("  Commands:")
            for cmd in content["commands"]:
                print(f"    $ {cmd}")

    print("\nEscalation Matrix:")
    for level, procedure in playbook["escalation"].items():
        print(f"  {level}: {procedure}")


# ---------------------------------------------------------------------------
# Exercise 4: Memory Forensics Guide
# ---------------------------------------------------------------------------

def exercise_4_memory_forensics():
    """Step-by-step guide for memory dump analysis with Volatility 3."""
    steps = [
        {
            "step": "1. List running processes",
            "command": "vol -f memory.dmp windows.pslist",
            "look_for": [
                "Processes with unusual names (misspelled system processes)",
                "svchost.exe running from non-System32 directory",
                "High PID processes that started recently",
                "Processes without a parent process",
            ],
        },
        {
            "step": "2. Find active network connections",
            "command": "vol -f memory.dmp windows.netscan",
            "look_for": [
                "Connections to known-bad IP addresses (check IOC database)",
                "Unusual outbound ports (4444, 8888, etc.)",
                "Processes making network connections that shouldn't",
                "Connections to countries not expected in business operations",
            ],
        },
        {
            "step": "3. Extract command line arguments",
            "command": "vol -f memory.dmp windows.cmdline",
            "look_for": [
                "PowerShell with -enc (encoded commands)",
                "cmd.exe with unusual arguments",
                "Processes launched from temp directories",
                "Arguments containing Base64-encoded strings",
            ],
        },
        {
            "step": "4. Identify injected code",
            "command": "vol -f memory.dmp windows.malfind",
            "look_for": [
                "Memory regions with PAGE_EXECUTE_READWRITE permissions",
                "Injected DLLs not on disk",
                "Shellcode patterns (NOP sleds, egg hunters)",
                "Processes with unexpected loaded modules",
            ],
        },
        {
            "step": "5. Recover credentials from memory",
            "command": "vol -f memory.dmp windows.hashdump",
            "look_for": [
                "NTLM hashes for local accounts",
                "Kerberos tickets (windows.kerberos)",
                "Browser-stored credentials",
                "SSH keys in process memory",
            ],
        },
    ]

    print("Memory Forensics Analysis Guide (Volatility 3)")
    print("=" * 60)
    for s in steps:
        print(f"\n{s['step']}")
        print(f"  Command: {s['command']}")
        print(f"  Indicators:")
        for item in s["look_for"]:
            print(f"    - {item}")


# ---------------------------------------------------------------------------
# Exercise 5: Post-Incident Report
# ---------------------------------------------------------------------------

@dataclass
class IncidentReport:
    incident_id: str
    title: str
    severity: str
    detected_at: str
    resolved_at: str
    summary: str
    timeline: list[dict]
    root_cause: str
    impact: str
    actions_taken: list[str]
    lessons_learned: list[str]
    follow_up_actions: list[dict]


def exercise_5_post_incident_report():
    """Generate a complete post-incident report."""
    report = IncidentReport(
        incident_id="INC-2025-001",
        title="Web Application Defacement via WordPress Plugin Exploit",
        severity="HIGH",
        detected_at="2025-01-15 03:15:00 UTC",
        resolved_at="2025-01-15 04:30:00 UTC",
        summary=(
            "Attacker exploited CVE-XXXX-XXXXX in the WP Gallery Plugin "
            "to deface the company homepage and create a backdoor admin account."
        ),
        timeline=[
            {"time": "03:00", "event": "Attacker exploits vulnerable WordPress plugin"},
            {"time": "03:02", "event": "Homepage replaced with political message"},
            {"time": "03:05", "event": "Backdoor admin account 'wp_backup' created"},
            {"time": "03:15", "event": "Monitoring system detects homepage content change"},
            {"time": "03:20", "event": "On-call engineer alerted via PagerDuty"},
            {"time": "03:30", "event": "Engineer confirms defacement and begins IR"},
            {"time": "03:35", "event": "WordPress admin access restricted to VPN"},
            {"time": "03:40", "event": "Backdoor account 'wp_backup' discovered and deleted"},
            {"time": "03:50", "event": "All admin passwords rotated"},
            {"time": "04:00", "event": "Homepage restored from backup"},
            {"time": "04:15", "event": "Vulnerable plugin removed"},
            {"time": "04:30", "event": "Incident resolved, monitoring confirmed stable"},
        ],
        root_cause=(
            "Unpatched WordPress plugin (WP Gallery v2.3.1) with a known "
            "remote code execution vulnerability (CVE published 2 weeks prior). "
            "The plugin was not included in the automated patch management system."
        ),
        impact=(
            "Public-facing homepage was defaced for approximately 1 hour. "
            "No customer data was accessed. A backdoor admin account was "
            "created but no evidence of data exfiltration was found."
        ),
        actions_taken=[
            "Blocked attacker IP at WAF level",
            "Removed backdoor admin account",
            "Rotated all WordPress admin credentials",
            "Restored homepage from known-good backup",
            "Removed vulnerable plugin",
            "Performed full WordPress security scan",
        ],
        lessons_learned=[
            "WordPress plugins were not included in vulnerability scanning",
            "Monitoring detected the change in 15 minutes — could be faster",
            "No WAF rule existed for this specific CVE",
            "Need a process for vetting WordPress plugins before installation",
        ],
        follow_up_actions=[
            {"action": "Add all WordPress plugins to vulnerability scanning",
             "owner": "Security Team", "due": "2025-01-22"},
            {"action": "Implement file integrity monitoring for web root",
             "owner": "DevOps", "due": "2025-01-29"},
            {"action": "Create WAF rules for common WordPress exploits",
             "owner": "Security Team", "due": "2025-01-22"},
            {"action": "Review and approve all installed WordPress plugins",
             "owner": "Security Team", "due": "2025-02-01"},
        ],
    )

    print("POST-INCIDENT REPORT")
    print("=" * 60)
    print(f"ID:       {report.incident_id}")
    print(f"Title:    {report.title}")
    print(f"Severity: {report.severity}")
    print(f"Detected: {report.detected_at}")
    print(f"Resolved: {report.resolved_at}")
    print(f"\nSummary: {report.summary}")

    print(f"\nTimeline:")
    for event in report.timeline:
        print(f"  {event['time']} — {event['event']}")

    print(f"\nRoot Cause: {report.root_cause}")
    print(f"\nImpact: {report.impact}")

    print(f"\nActions Taken:")
    for action in report.actions_taken:
        print(f"  - {action}")

    print(f"\nLessons Learned:")
    for lesson in report.lessons_learned:
        print(f"  - {lesson}")

    print(f"\nFollow-up Actions:")
    for action in report.follow_up_actions:
        print(f"  - [{action['due']}] {action['action']} (Owner: {action['owner']})")


# ---------------------------------------------------------------------------
# Exercise 6: Network Analysis (PCAP Methodology)
# ---------------------------------------------------------------------------

def exercise_6_network_analysis():
    """Guide for PCAP analysis methodology."""
    print("PCAP Analysis Methodology")
    print("=" * 60)
    print("""
Step 1: Overview Statistics
  $ capinfos capture.pcap
  $ tshark -r capture.pcap -q -z conv,ip

Step 2: Top Talkers
  $ tshark -r capture.pcap -q -z endpoints,ip
  Look for: IPs with disproportionate traffic volume

Step 3: DNS Analysis
  $ tshark -r capture.pcap -Y dns -T fields -e dns.qry.name | sort | uniq -c | sort -rn
  Look for:
    - Queries to known malicious domains
    - DNS tunneling (unusually long subdomains)
    - High-frequency lookups to single domain

Step 4: HTTP Analysis
  $ tshark -r capture.pcap -Y http -T fields \\
      -e ip.src -e http.request.method -e http.host -e http.request.uri
  Look for:
    - Downloads of executables (.exe, .dll, .ps1)
    - POST requests to unusual endpoints (C2 communication)
    - User-Agent strings associated with malware
    - Beaconing patterns (regular intervals)

Step 5: Data Exfiltration Indicators
  $ tshark -r capture.pcap -q -z io,stat,60
  Look for:
    - Large outbound data transfers
    - Encrypted traffic to unknown destinations
    - DNS queries with encoded data (Base64 in subdomains)
    - HTTP POST with large payloads to external servers

Step 6: Generate Report
  Structure:
    1. Executive Summary
    2. Scope and Methodology
    3. Top Talkers (IP addresses, traffic volume)
    4. DNS Findings (suspicious queries)
    5. HTTP Findings (malware, C2)
    6. Indicators of Compromise (IPs, domains, hashes)
    7. Recommendations
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Log Analysis")
    print("=" * 70)
    exercise_1_log_analysis()

    print("\n" + "=" * 70)
    print("Exercise 2: IOC Database")
    print("=" * 70)
    exercise_2_ioc_database()

    print("\n" + "=" * 70)
    print("Exercise 3: IR Playbook (SQL Injection)")
    print("=" * 70)
    exercise_3_ir_playbook()

    print("\n" + "=" * 70)
    print("Exercise 4: Memory Forensics Guide")
    print("=" * 70)
    exercise_4_memory_forensics()

    print("\n" + "=" * 70)
    print("Exercise 5: Post-Incident Report")
    print("=" * 70)
    exercise_5_post_incident_report()

    print("\n" + "=" * 70)
    print("Exercise 6: Network Analysis Methodology")
    print("=" * 70)
    exercise_6_network_analysis()
