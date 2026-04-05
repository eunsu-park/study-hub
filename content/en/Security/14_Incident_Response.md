# Incident Response and Forensics

**Previous**: [13. Security Testing](13_Security_Testing.md) | **Next**: [15. Project: Building a Secure REST API](15_Project_Secure_API.md)

---

Security incidents are inevitable. What separates resilient organizations from vulnerable ones is not whether they get breached, but how effectively they respond. This lesson covers the complete incident response lifecycle based on the NIST framework, digital forensics fundamentals, log analysis techniques, and practical Python scripts for detecting indicators of compromise (IOCs). By the end, you will be able to build incident response playbooks and analyze security events systematically.

## Learning Objectives

- Understand the NIST Incident Response lifecycle phases
- Build and maintain an incident response plan
- Analyze logs for indicators of compromise
- Understand digital forensics fundamentals and chain of custody
- Write Python scripts for log parsing and IOC detection
- Create incident response playbooks for common scenarios
- Conduct effective post-incident reviews

---

## 1. Incident Response Overview

### 1.1 What is a Security Incident?

A security incident is any event that compromises the confidentiality, integrity, or availability of information or systems. Not every security event is an incident -- triage determines severity.

```
┌─────────────────────────────────────────────────────────────────┐
│              Security Event → Incident Classification            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Security Events (millions/day)                                  │
│  ├── Firewall blocks                                            │
│  ├── Failed login attempts                                      │
│  ├── Port scans                                                  │
│  ├── Malware detections (quarantined)                           │
│  └── IDS/IPS alerts                                             │
│       │                                                          │
│       ▼  Triage & Correlation                                    │
│                                                                  │
│  Security Incidents (few/month)                                  │
│  ├── Successful unauthorized access                             │
│  ├── Data breach / exfiltration                                 │
│  ├── Malware infection (active)                                 │
│  ├── Denial of service attack                                   │
│  ├── Insider threat activity                                    │
│  └── Ransomware deployment                                      │
│       │                                                          │
│       ▼  Severity Classification                                 │
│                                                                  │
│  ┌───────────┬──────────────────────────────────────────────┐   │
│  │ Severity  │ Description                                  │   │
│  ├───────────┼──────────────────────────────────────────────┤   │
│  │ Critical  │ Active data breach, ransomware, system-wide  │   │
│  │ (P1)      │ compromise. Immediate response required.     │   │
│  ├───────────┼──────────────────────────────────────────────┤   │
│  │ High      │ Confirmed intrusion, single system           │   │
│  │ (P2)      │ compromised, active malware. Hours.          │   │
│  ├───────────┼──────────────────────────────────────────────┤   │
│  │ Medium    │ Suspicious activity, policy violation,       │   │
│  │ (P3)      │ vulnerability actively exploited. Days.      │   │
│  ├───────────┼──────────────────────────────────────────────┤   │
│  │ Low       │ Minor policy violation, unsuccessful         │   │
│  │ (P4)      │ attack attempt, informational. Weeks.        │   │
│  └───────────┴──────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Common Incident Types

| Incident Type | Examples | Typical Indicators |
|---|---|---|
| **Malware** | Ransomware, trojan, worm | Unusual processes, file encryption, C2 traffic |
| **Unauthorized Access** | Compromised credentials, brute force | Failed logins, off-hours access, unusual geolocations |
| **Data Breach** | Exfiltration, accidental exposure | Large data transfers, DB dumps, unusual queries |
| **DoS/DDoS** | Volumetric, application-layer | Traffic spikes, service degradation, CPU exhaustion |
| **Insider Threat** | Data theft, sabotage | Excessive access, large downloads, policy violations |
| **Web Application** | SQLi, XSS, RCE | Suspicious request patterns, WAF alerts, error spikes |
| **Supply Chain** | Compromised dependency, update | Unexpected package changes, suspicious build artifacts |
| **Phishing** | Credential harvesting, BEC | Reported emails, unusual login locations, wire transfer requests |

---

## 2. NIST Incident Response Lifecycle

### 2.1 The Four Phases

The NIST Computer Security Incident Handling Guide (SP 800-61) defines four main phases. In practice, these phases overlap and cycle.

```
┌─────────────────────────────────────────────────────────────────┐
│              NIST Incident Response Lifecycle                     │
│              (SP 800-61 Rev. 2)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┐                                           │
│   │  1. Preparation  │◄──────────────────────────────┐          │
│   │                   │                               │          │
│   │  - IR plan        │                               │          │
│   │  - Team training  │                               │          │
│   │  - Tools ready    │                               │          │
│   │  - Communication  │                               │          │
│   └────────┬──────────┘                               │          │
│            │                                          │          │
│            ▼                                          │          │
│   ┌─────────────────────────┐                        │          │
│   │  2. Detection &          │                        │          │
│   │     Analysis             │                        │          │
│   │                          │                        │          │
│   │  - Monitor alerts        │                        │          │
│   │  - Triage events         │                        │          │
│   │  - Determine scope       │                        │          │
│   │  - Classify severity     │                        │          │
│   └────────┬────────────────┘                        │          │
│            │                                          │          │
│            ▼                                          │          │
│   ┌─────────────────────────┐                        │          │
│   │  3. Containment,        │    ◄── May cycle       │          │
│   │     Eradication &       │        between these    │          │
│   │     Recovery            │        sub-phases       │          │
│   │                          │                        │          │
│   │  - Isolate affected     │                        │          │
│   │  - Remove threat        │                        │          │
│   │  - Restore systems      │                        │          │
│   │  - Validate recovery    │                        │          │
│   └────────┬────────────────┘                        │          │
│            │                                          │          │
│            ▼                                          │          │
│   ┌─────────────────────────┐                        │          │
│   │  4. Post-Incident       │────────────────────────┘          │
│   │     Activity            │    Lessons feed back              │
│   │                          │    into Preparation               │
│   │  - Lessons learned      │                                    │
│   │  - Report writing       │                                    │
│   │  - Process improvement  │                                    │
│   └─────────────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Phase 1: Preparation

Preparation is the most critical phase. Without it, everything else is improvisation under pressure.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Preparation Checklist                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  People:                                                          │
│  [x] IR team identified with clear roles                         │
│  [x] Contact list (team, management, legal, PR, vendors)        │
│  [x] On-call rotation schedule                                   │
│  [x] Regular training and tabletop exercises                     │
│  [x] External IR retainer (optional)                             │
│                                                                   │
│  Process:                                                         │
│  [x] Written IR plan approved by management                     │
│  [x] Playbooks for common incident types                        │
│  [x] Escalation procedures                                       │
│  [x] Communication templates (internal, external, legal)        │
│  [x] Evidence handling procedures                                │
│  [x] Regulatory notification requirements documented             │
│                                                                   │
│  Technology:                                                      │
│  [x] Logging infrastructure (centralized, retained)             │
│  [x] SIEM or log analysis tools                                 │
│  [x] Forensic workstation / toolkit                              │
│  [x] Network monitoring / IDS                                    │
│  [x] Endpoint detection and response (EDR)                      │
│  [x] Backup and recovery systems tested                         │
│  [x] Clean OS images / golden images                            │
│  [x] Jump bag: portable forensic tools, cables, storage         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

#### IR Team Roles

```
┌──────────────────────────────────────────────────────────────────┐
│                   Incident Response Team Roles                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐                                         │
│  │   Incident Manager  │  Overall coordination, decisions,       │
│  │   (Team Lead)       │  communication with management          │
│  └─────────┬───────────┘                                         │
│            │                                                      │
│  ┌─────────┴────────────────────────────────┐                    │
│  │         │              │                  │                    │
│  ▼         ▼              ▼                  ▼                    │
│ ┌────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐            │
│ │Security│ │  Forensic  │ │  System  │ │ Comms /  │            │
│ │Analyst │ │  Analyst   │ │  Admin   │ │  Legal   │            │
│ │        │ │            │ │          │ │          │            │
│ │Monitor │ │Evidence    │ │Contain & │ │Notify &  │            │
│ │Triage  │ │Collection  │ │Recover   │ │Document  │            │
│ │Analyze │ │Analysis    │ │Patch     │ │Regulate  │            │
│ └────────┘ └────────────┘ └──────────┘ └──────────┘            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 Phase 2: Detection and Analysis

#### Detection Sources

```
┌──────────────────────────────────────────────────────────────────┐
│                    Detection Sources                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Automated:                                                       │
│  ├── SIEM alerts (correlated log events)                         │
│  ├── IDS/IPS (Snort, Suricata)                                  │
│  ├── EDR alerts (CrowdStrike, Carbon Black, etc.)               │
│  ├── WAF alerts (ModSecurity, AWS WAF)                          │
│  ├── Antivirus / Anti-malware                                    │
│  ├── File integrity monitoring (OSSEC, Tripwire)                │
│  ├── Network traffic anomalies (NetFlow, Zeek)                  │
│  └── Cloud security alerts (GuardDuty, Security Center)         │
│                                                                   │
│  Human:                                                           │
│  ├── User reports ("something looks wrong")                      │
│  ├── Help desk tickets                                           │
│  ├── External notification (partner, vendor, researcher)        │
│  ├── Law enforcement notification                                │
│  ├── Media reports                                               │
│  └── Threat intelligence feeds                                   │
│                                                                   │
│  Proactive:                                                       │
│  ├── Threat hunting                                              │
│  ├── Penetration testing results                                 │
│  ├── Vulnerability scanning                                      │
│  └── Log review / audit                                          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

#### Initial Analysis Questions

When an alert fires, systematically answer these questions:

```
1. WHAT happened?
   - What systems/data are affected?
   - What type of incident is this?
   - What is the initial evidence?

2. WHEN did it happen?
   - When was the first indicator?
   - When was it detected?
   - Is it ongoing?

3. WHERE is the impact?
   - Which hosts/networks?
   - Which applications/services?
   - Which data/users?

4. WHO is involved?
   - Source IPs/accounts?
   - Targeted users/systems?
   - Internal or external actor?

5. HOW did it happen?
   - Attack vector (phishing, exploit, insider, etc.)?
   - Vulnerability exploited?
   - Tools/techniques used?

6. HOW BAD is it?
   - Scope: how many systems affected?
   - Impact: data loss, service disruption?
   - Severity classification (P1-P4)?
```

### 2.4 Phase 3: Containment, Eradication, and Recovery

```
┌──────────────────────────────────────────────────────────────────┐
│          Containment → Eradication → Recovery                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────┐                 │
│  │  CONTAINMENT (Stop the Bleeding)             │                 │
│  │                                              │                 │
│  │  Short-term:                                 │                 │
│  │  ├── Isolate affected systems (network)     │                 │
│  │  ├── Block malicious IPs/domains            │                 │
│  │  ├── Disable compromised accounts           │                 │
│  │  ├── Redirect DNS if needed                 │                 │
│  │  └── Preserve evidence before changes       │                 │
│  │                                              │                 │
│  │  Long-term:                                  │                 │
│  │  ├── Apply temporary patches/workarounds    │                 │
│  │  ├── Increase monitoring on affected area   │                 │
│  │  ├── Implement additional access controls   │                 │
│  │  └── Set up honeypot/canary if appropriate  │                 │
│  └─────────────────────────────────────────────┘                 │
│            │                                                      │
│            ▼                                                      │
│  ┌─────────────────────────────────────────────┐                 │
│  │  ERADICATION (Remove the Threat)             │                 │
│  │                                              │                 │
│  │  ├── Remove malware / backdoors             │                 │
│  │  ├── Close vulnerability that was exploited │                 │
│  │  ├── Reset compromised credentials          │                 │
│  │  ├── Rebuild affected systems if needed     │                 │
│  │  ├── Update firewall/IDS rules              │                 │
│  │  └── Scan for persistence mechanisms        │                 │
│  └─────────────────────────────────────────────┘                 │
│            │                                                      │
│            ▼                                                      │
│  ┌─────────────────────────────────────────────┐                 │
│  │  RECOVERY (Return to Normal)                 │                 │
│  │                                              │                 │
│  │  ├── Restore from clean backups             │                 │
│  │  ├── Rebuild systems from golden images     │                 │
│  │  ├── Gradually restore services             │                 │
│  │  ├── Verify system integrity                │                 │
│  │  ├── Monitor closely for re-compromise      │                 │
│  │  └── Declare incident resolved              │                 │
│  └─────────────────────────────────────────────┘                 │
│                                                                   │
│  IMPORTANT: Document every action with timestamps!               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.5 Phase 4: Post-Incident Activity (Lessons Learned)

```
┌──────────────────────────────────────────────────────────────────┐
│              Post-Incident Review Meeting Agenda                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Schedule: Within 1-2 weeks of incident closure                  │
│  Attendees: All involved parties (blameless environment)         │
│                                                                   │
│  1. Timeline Review (30 min)                                     │
│     - Walk through the complete timeline                         │
│     - When was first indicator? When detected? When resolved?   │
│                                                                   │
│  2. What Went Well (15 min)                                      │
│     - Effective detection mechanisms                              │
│     - Quick containment actions                                   │
│     - Good communication                                         │
│                                                                   │
│  3. What Could Be Improved (30 min)                              │
│     - Detection gaps                                              │
│     - Response delays                                             │
│     - Communication breakdowns                                    │
│     - Tool/process gaps                                           │
│                                                                   │
│  4. Root Cause Analysis (30 min)                                 │
│     - What was the root cause?                                   │
│     - Why did existing controls fail?                            │
│     - Use "5 Whys" technique                                     │
│                                                                   │
│  5. Action Items (15 min)                                        │
│     - Specific improvements with owners and deadlines            │
│     - Process changes                                             │
│     - Technology changes                                          │
│     - Training needs                                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Log Analysis and SIEM Concepts

### 3.1 Logging Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  Centralized Logging Architecture                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Log Sources                    Collection        Storage/Analysis│
│  ┌──────────┐                                                     │
│  │ Web      │──┐                                                  │
│  │ Servers  │  │    ┌───────────────┐    ┌───────────────────┐   │
│  └──────────┘  ├──► │  Log Shipper  │──► │                   │   │
│  ┌──────────┐  │    │  (Filebeat,   │    │  SIEM / Log       │   │
│  │ App      │──┤    │   Fluentd,    │    │  Management       │   │
│  │ Servers  │  │    │   rsyslog)    │    │                   │   │
│  └──────────┘  │    └───────────────┘    │  - Elasticsearch  │   │
│  ┌──────────┐  │                         │  - Splunk         │   │
│  │ Database │──┤    ┌───────────────┐    │  - Graylog        │   │
│  │ Servers  │  ├──► │  Message      │──► │  - Wazuh          │   │
│  └──────────┘  │    │  Queue        │    │  - QRadar         │   │
│  ┌──────────┐  │    │  (Kafka,      │    │                   │   │
│  │ Firewall │──┤    │   Redis)      │    │  Features:        │   │
│  │ / IDS    │  │    └───────────────┘    │  - Search         │   │
│  └──────────┘  │                         │  - Correlation    │   │
│  ┌──────────┐  │                         │  - Alerting       │   │
│  │ Endpoint │──┘                         │  - Dashboards     │   │
│  │ Agents   │                            │  - Retention      │   │
│  └──────────┘                            └───────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Critical Log Sources

| Log Source | What to Capture | Security Value |
|---|---|---|
| **Web server** (nginx, Apache) | Access logs, error logs | Attack detection, anomaly detection |
| **Application** | Auth events, errors, API calls | Business logic attacks, abuse |
| **Database** | Queries, connections, errors | SQL injection, data exfiltration |
| **OS / System** | Auth logs, process exec, file changes | Privilege escalation, persistence |
| **Firewall** | Allow/deny, connections | Network attacks, lateral movement |
| **DNS** | Queries, responses | C2 communication, data exfiltration |
| **Email** | Send/receive, attachments | Phishing, data exfiltration |
| **Cloud** | API calls, config changes | Misconfiguration, unauthorized access |

### 3.3 SIEM Correlation Rules

```
┌──────────────────────────────────────────────────────────────────┐
│                   Common SIEM Correlation Rules                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Rule 1: Brute Force Detection                                   │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: >10 failed logins from same IP in 5 minutes    │          │
│  │ THEN: Alert "Possible brute force attack"          │          │
│  │ SEVERITY: Medium                                    │          │
│  │ ACTION: Block IP temporarily, notify SOC            │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  Rule 2: Impossible Travel                                       │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: Same user logs in from 2 geolocations          │          │
│  │     that are >500 miles apart within 30 minutes    │          │
│  │ THEN: Alert "Impossible travel detected"           │          │
│  │ SEVERITY: High                                      │          │
│  │ ACTION: Force re-authentication, notify user        │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  Rule 3: Data Exfiltration                                       │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: >100MB data transfer to external IP             │          │
│  │     from a server that normally sends <1MB/hour    │          │
│  │ THEN: Alert "Possible data exfiltration"           │          │
│  │ SEVERITY: Critical                                  │          │
│  │ ACTION: Block transfer, isolate host, alert IR      │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  Rule 4: Privilege Escalation                                    │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: User added to admin/root group                  │          │
│  │     AND change was not from approved change system │          │
│  │ THEN: Alert "Unauthorized privilege escalation"    │          │
│  │ SEVERITY: Critical                                  │          │
│  │ ACTION: Revert change, disable account, alert IR    │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  Rule 5: Web Application Attack                                  │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: >5 WAF blocks from same IP in 1 minute         │          │
│  │     AND HTTP 500 errors increase from same app     │          │
│  │ THEN: Alert "Active web application attack"        │          │
│  │ SEVERITY: High                                      │          │
│  │ ACTION: Block IP, increase logging, alert IR        │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Python: Log Parsing and Analysis

### 4.1 Web Server Log Parser

```python
"""
log_parser.py - Parse and analyze web server access logs.
Supports Apache/Nginx combined log format.

Example log line:
192.168.1.100 - admin [15/Jan/2025:10:30:45 +0000] "GET /admin HTTP/1.1" 200 1234 "http://example.com" "Mozilla/5.0"
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator


# Combined log format regex
LOG_PATTERN = re.compile(
    r'(?P<ip>\S+)\s+'              # IP address
    r'\S+\s+'                       # ident (usually -)
    r'(?P<user>\S+)\s+'            # authenticated user
    r'\[(?P<time>[^\]]+)\]\s+'     # timestamp
    r'"(?P<method>\S+)\s+'         # HTTP method
    r'(?P<path>\S+)\s+'            # request path
    r'(?P<protocol>\S+)"\s+'       # HTTP version
    r'(?P<status>\d+)\s+'          # status code
    r'(?P<size>\S+)\s+'            # response size
    r'"(?P<referer>[^"]*)"\s+'     # referer
    r'"(?P<agent>[^"]*)"'          # user agent
)


@dataclass
class LogEntry:
    """Parsed log entry."""
    ip: str
    user: str
    timestamp: datetime
    method: str
    path: str
    protocol: str
    status: int
    size: int
    referer: str
    user_agent: str
    raw: str = ""


@dataclass
class SecurityAlert:
    """A security alert generated from log analysis."""
    alert_type: str
    severity: str           # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    source_ip: Optional[str] = None
    count: int = 0
    sample_entries: list = field(default_factory=list)
    timestamp: str = ""

    def __str__(self):
        return (f"[{self.severity}] {self.alert_type}: {self.description} "
                f"(IP: {self.source_ip}, count: {self.count})")


def parse_log_line(line: str) -> Optional[LogEntry]:
    """Parse a single log line into a LogEntry."""
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None

    data = match.groupdict()

    # Parse timestamp
    try:
        ts = datetime.strptime(data['time'], '%d/%b/%Y:%H:%M:%S %z')
    except ValueError:
        ts = datetime.now()

    # Parse size (may be '-')
    try:
        size = int(data['size'])
    except ValueError:
        size = 0

    return LogEntry(
        ip=data['ip'],
        user=data['user'],
        timestamp=ts,
        method=data['method'],
        path=data['path'],
        protocol=data['protocol'],
        status=int(data['status']),
        size=size,
        referer=data['referer'],
        user_agent=data['agent'],
        raw=line.strip()
    )


def parse_log_file(filepath: str) -> Iterator[LogEntry]:
    """Parse all entries from a log file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    with path.open('r', errors='ignore') as f:
        for line in f:
            entry = parse_log_line(line)
            if entry:
                yield entry


class LogAnalyzer:
    """Analyze parsed log entries for security indicators."""

    def __init__(self):
        self.entries: list[LogEntry] = []
        self.alerts: list[SecurityAlert] = []

    def load(self, filepath: str) -> int:
        """Load and parse a log file. Returns entry count."""
        self.entries = list(parse_log_file(filepath))
        return len(self.entries)

    def analyze_all(self) -> list[SecurityAlert]:
        """Run all analysis rules."""
        self.alerts = []
        self.detect_brute_force()
        self.detect_directory_traversal()
        self.detect_sql_injection()
        self.detect_scanner_activity()
        self.detect_error_spikes()
        self.detect_suspicious_user_agents()
        self.detect_admin_access()
        return self.alerts

    def detect_brute_force(self, threshold: int = 10,
                            window_minutes: int = 5) -> None:
        """Detect brute force login attempts."""
        # Group 401/403 responses by IP
        failed_logins = defaultdict(list)

        for entry in self.entries:
            if entry.status in (401, 403):
                failed_logins[entry.ip].append(entry)

        for ip, entries in failed_logins.items():
            if len(entries) >= threshold:
                # Check if they occur within the time window
                entries.sort(key=lambda e: e.timestamp)
                for i in range(len(entries) - threshold + 1):
                    window = entries[i:i + threshold]
                    time_diff = (window[-1].timestamp -
                                 window[0].timestamp).total_seconds()
                    if time_diff <= window_minutes * 60:
                        self.alerts.append(SecurityAlert(
                            alert_type="BRUTE_FORCE",
                            severity="HIGH",
                            description=(
                                f"{len(entries)} failed auth attempts from {ip} "
                                f"({threshold}+ within {window_minutes} min)"
                            ),
                            source_ip=ip,
                            count=len(entries),
                            sample_entries=[e.raw for e in entries[:3]],
                        ))
                        break  # One alert per IP

    def detect_directory_traversal(self) -> None:
        """Detect path traversal attempts."""
        traversal_patterns = [
            '../', '..\\', '%2e%2e', '%252e%252e',
            '/etc/passwd', '/etc/shadow', '/windows/system32',
            'boot.ini', 'web.config',
        ]

        traversal_attempts = defaultdict(list)

        for entry in self.entries:
            path_lower = entry.path.lower()
            for pattern in traversal_patterns:
                if pattern in path_lower:
                    traversal_attempts[entry.ip].append(entry)
                    break

        for ip, entries in traversal_attempts.items():
            self.alerts.append(SecurityAlert(
                alert_type="DIRECTORY_TRAVERSAL",
                severity="HIGH",
                description=(
                    f"Path traversal attempt from {ip}: "
                    f"{entries[0].path}"
                ),
                source_ip=ip,
                count=len(entries),
                sample_entries=[e.raw for e in entries[:3]],
            ))

    def detect_sql_injection(self) -> None:
        """Detect SQL injection attempts in request paths."""
        sqli_patterns = [
            "' or ", "' and ", "union select", "order by",
            "1=1", "' --", "'; drop", "sleep(",
            "benchmark(", "waitfor delay", "pg_sleep",
            "%27", "char(", "concat(",
        ]

        sqli_attempts = defaultdict(list)

        for entry in self.entries:
            path_lower = entry.path.lower()
            for pattern in sqli_patterns:
                if pattern in path_lower:
                    sqli_attempts[entry.ip].append(entry)
                    break

        for ip, entries in sqli_attempts.items():
            self.alerts.append(SecurityAlert(
                alert_type="SQL_INJECTION",
                severity="CRITICAL",
                description=(
                    f"SQL injection attempt from {ip}: "
                    f"{entries[0].path[:100]}"
                ),
                source_ip=ip,
                count=len(entries),
                sample_entries=[e.raw for e in entries[:3]],
            ))

    def detect_scanner_activity(self) -> None:
        """Detect automated vulnerability scanner activity."""
        scanner_paths = [
            '/.env', '/wp-admin', '/wp-login.php',
            '/phpmyadmin', '/admin', '/administrator',
            '/.git/config', '/.svn/entries',
            '/robots.txt', '/sitemap.xml',
            '/backup', '/database', '/db',
            '/server-status', '/server-info',
            '/.htaccess', '/web.config',
            '/xmlrpc.php', '/api/v1',
        ]

        ip_scanner_hits = defaultdict(set)

        for entry in self.entries:
            for scanner_path in scanner_paths:
                if entry.path.lower().startswith(scanner_path):
                    ip_scanner_hits[entry.ip].add(entry.path)

        for ip, paths in ip_scanner_hits.items():
            if len(paths) >= 5:  # Hit 5+ scanner paths
                self.alerts.append(SecurityAlert(
                    alert_type="VULNERABILITY_SCANNER",
                    severity="MEDIUM",
                    description=(
                        f"Scanner activity from {ip}: "
                        f"probed {len(paths)} common paths"
                    ),
                    source_ip=ip,
                    count=len(paths),
                    sample_entries=list(paths)[:5],
                ))

    def detect_error_spikes(self, threshold: int = 50) -> None:
        """Detect unusual spikes in error responses."""
        # Count 5xx errors per IP
        error_counts = Counter()
        for entry in self.entries:
            if 500 <= entry.status < 600:
                error_counts[entry.ip] += 1

        for ip, count in error_counts.most_common():
            if count >= threshold:
                self.alerts.append(SecurityAlert(
                    alert_type="ERROR_SPIKE",
                    severity="MEDIUM",
                    description=(
                        f"High error rate from {ip}: "
                        f"{count} server errors (5xx)"
                    ),
                    source_ip=ip,
                    count=count,
                ))

    def detect_suspicious_user_agents(self) -> None:
        """Detect requests with suspicious user agents."""
        suspicious_agents = [
            'sqlmap', 'nikto', 'nmap', 'masscan',
            'dirbuster', 'gobuster', 'wfuzz',
            'burpsuite', 'acunetix', 'nessus',
            'python-requests',  # May be legitimate, but flag
            'curl/',            # May be legitimate, but flag
        ]

        for entry in self.entries:
            agent_lower = entry.user_agent.lower()
            for sus_agent in suspicious_agents:
                if sus_agent in agent_lower:
                    self.alerts.append(SecurityAlert(
                        alert_type="SUSPICIOUS_USER_AGENT",
                        severity="MEDIUM",
                        description=(
                            f"Suspicious user agent from {entry.ip}: "
                            f"{entry.user_agent[:80]}"
                        ),
                        source_ip=entry.ip,
                        count=1,
                    ))
                    break  # One alert per entry

    def detect_admin_access(self) -> None:
        """Detect access to administrative endpoints."""
        admin_paths = ['/admin', '/dashboard', '/manage', '/api/admin']
        admin_access = defaultdict(list)

        for entry in self.entries:
            for admin_path in admin_paths:
                if entry.path.lower().startswith(admin_path):
                    if entry.status == 200:
                        admin_access[entry.ip].append(entry)

        for ip, entries in admin_access.items():
            self.alerts.append(SecurityAlert(
                alert_type="ADMIN_ACCESS",
                severity="LOW",
                description=(
                    f"Successful admin access from {ip}: "
                    f"{entries[0].path}"
                ),
                source_ip=ip,
                count=len(entries),
            ))

    def print_report(self) -> None:
        """Print a formatted analysis report."""
        print("=" * 65)
        print("  LOG ANALYSIS SECURITY REPORT")
        print("=" * 65)
        print(f"  Total entries analyzed: {len(self.entries)}")
        print(f"  Total alerts: {len(self.alerts)}")

        severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        for severity in severity_order:
            alerts = [a for a in self.alerts if a.severity == severity]
            if not alerts:
                continue

            print(f"\n{'─' * 65}")
            print(f"  {severity} ({len(alerts)})")
            print(f"{'─' * 65}")

            for alert in alerts:
                print(f"\n  [{alert.alert_type}]")
                print(f"  {alert.description}")
                if alert.sample_entries:
                    print(f"  Samples:")
                    for sample in alert.sample_entries[:2]:
                        print(f"    {str(sample)[:100]}")

        # Summary statistics
        print(f"\n{'─' * 65}")
        print("  TOP SOURCE IPs")
        print(f"{'─' * 65}")
        ip_alert_count = Counter()
        for alert in self.alerts:
            if alert.source_ip:
                ip_alert_count[alert.source_ip] += 1
        for ip, count in ip_alert_count.most_common(10):
            print(f"  {ip:20s} {count} alerts")

        print(f"\n{'=' * 65}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <access.log>")
        sys.exit(1)

    analyzer = LogAnalyzer()
    count = analyzer.load(sys.argv[1])
    print(f"[*] Loaded {count} log entries")

    alerts = analyzer.analyze_all()
    analyzer.print_report()
```

### 4.2 IOC Detection Script

```python
"""
ioc_detector.py - Indicator of Compromise (IOC) detection.
Checks files, network connections, and system state for known IOCs.
"""

import hashlib
import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class IOC:
    """An Indicator of Compromise."""
    ioc_type: str       # IP, DOMAIN, HASH_MD5, HASH_SHA256, FILENAME, REGEX
    value: str
    description: str = ""
    source: str = ""    # Where this IOC came from
    severity: str = "MEDIUM"


@dataclass
class IOCMatch:
    """A match found during scanning."""
    ioc: IOC
    location: str       # Where the match was found
    context: str = ""   # Additional context
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class IOCDatabase:
    """
    Simple IOC database.
    In production, use STIX/TAXII or a proper threat intelligence platform.
    """

    def __init__(self):
        self.iocs: list[IOC] = []

    def load_from_json(self, filepath: str) -> int:
        """Load IOCs from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        for item in data.get('iocs', []):
            self.iocs.append(IOC(
                ioc_type=item['type'],
                value=item['value'],
                description=item.get('description', ''),
                source=item.get('source', ''),
                severity=item.get('severity', 'MEDIUM'),
            ))

        return len(self.iocs)

    def load_sample_iocs(self) -> None:
        """Load sample IOCs for demonstration."""
        # These are FAKE IOCs for educational purposes
        sample_iocs = [
            IOC("IP", "198.51.100.1", "Known C2 server", "sample", "HIGH"),
            IOC("IP", "203.0.113.50", "Phishing infrastructure", "sample", "HIGH"),
            IOC("DOMAIN", "evil-malware.example.com", "Malware C2", "sample", "CRITICAL"),
            IOC("DOMAIN", "phish.example.net", "Phishing domain", "sample", "HIGH"),
            IOC("HASH_SHA256",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "Known malware hash (empty file SHA256)", "sample", "MEDIUM"),
            IOC("FILENAME", "mimikatz.exe", "Credential dumping tool", "sample", "CRITICAL"),
            IOC("FILENAME", "nc.exe", "Netcat - possible backdoor", "sample", "HIGH"),
            IOC("REGEX", r"(?:eval|exec)\s*\(\s*base64", "Obfuscated code execution",
                "sample", "HIGH"),
        ]
        self.iocs.extend(sample_iocs)

    def get_by_type(self, ioc_type: str) -> list[IOC]:
        """Get all IOCs of a specific type."""
        return [ioc for ioc in self.iocs if ioc.ioc_type == ioc_type]


class IOCScanner:
    """Scan system for Indicators of Compromise."""

    def __init__(self, ioc_db: IOCDatabase):
        self.ioc_db = ioc_db
        self.matches: list[IOCMatch] = []

    def scan_all(self, scan_dir: Optional[str] = None) -> list[IOCMatch]:
        """Run all IOC scans."""
        self.matches = []
        print("[*] Starting IOC scan...")

        self.scan_file_hashes(scan_dir or "/tmp")
        self.scan_file_names(scan_dir or "/tmp")
        self.scan_network_connections()
        self.scan_dns_cache()
        self.scan_file_contents(scan_dir or "/tmp")

        return self.matches

    def scan_file_hashes(self, directory: str) -> None:
        """Check file hashes against known malware hashes."""
        print(f"[*] Scanning file hashes in {directory}...")
        hash_iocs = {
            ioc.value.lower(): ioc
            for ioc in self.ioc_db.get_by_type("HASH_SHA256")
        }
        md5_iocs = {
            ioc.value.lower(): ioc
            for ioc in self.ioc_db.get_by_type("HASH_MD5")
        }

        if not hash_iocs and not md5_iocs:
            return

        scan_path = Path(directory)
        for filepath in scan_path.rglob("*"):
            if not filepath.is_file():
                continue
            # Skip very large files (>100MB)
            try:
                if filepath.stat().st_size > 100 * 1024 * 1024:
                    continue
            except OSError:
                continue

            try:
                content = filepath.read_bytes()
                sha256 = hashlib.sha256(content).hexdigest().lower()
                md5 = hashlib.md5(content).hexdigest().lower()

                if sha256 in hash_iocs:
                    self.matches.append(IOCMatch(
                        ioc=hash_iocs[sha256],
                        location=str(filepath),
                        context=f"SHA256: {sha256}",
                    ))

                if md5 in md5_iocs:
                    self.matches.append(IOCMatch(
                        ioc=md5_iocs[md5],
                        location=str(filepath),
                        context=f"MD5: {md5}",
                    ))

            except (PermissionError, OSError):
                pass

    def scan_file_names(self, directory: str) -> None:
        """Check for files with known malicious names."""
        print(f"[*] Scanning file names in {directory}...")
        name_iocs = {
            ioc.value.lower(): ioc
            for ioc in self.ioc_db.get_by_type("FILENAME")
        }

        if not name_iocs:
            return

        scan_path = Path(directory)
        for filepath in scan_path.rglob("*"):
            if filepath.name.lower() in name_iocs:
                self.matches.append(IOCMatch(
                    ioc=name_iocs[filepath.name.lower()],
                    location=str(filepath),
                    context=f"Filename match: {filepath.name}",
                ))

    def scan_network_connections(self) -> None:
        """Check active network connections against known bad IPs."""
        print("[*] Scanning network connections...")
        ip_iocs = {
            ioc.value: ioc
            for ioc in self.ioc_db.get_by_type("IP")
        }

        if not ip_iocs:
            return

        try:
            # Use netstat or ss to get connections
            result = subprocess.run(
                ["netstat", "-an"],
                capture_output=True, text=True, timeout=10
            )

            for line in result.stdout.splitlines():
                for bad_ip in ip_iocs:
                    if bad_ip in line:
                        self.matches.append(IOCMatch(
                            ioc=ip_iocs[bad_ip],
                            location="Active network connection",
                            context=line.strip(),
                        ))

        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Try ss as fallback
            try:
                result = subprocess.run(
                    ["ss", "-an"],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    for bad_ip in ip_iocs:
                        if bad_ip in line:
                            self.matches.append(IOCMatch(
                                ioc=ip_iocs[bad_ip],
                                location="Active network connection",
                                context=line.strip(),
                            ))
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print("    [!] Could not check network connections")

    def scan_dns_cache(self) -> None:
        """Check DNS resolutions for known malicious domains."""
        print("[*] Checking known malicious domains...")
        domain_iocs = self.ioc_db.get_by_type("DOMAIN")

        for ioc in domain_iocs:
            try:
                # Try to resolve the domain - if it resolves from cache
                # the system may have contacted it
                result = socket.getaddrinfo(
                    ioc.value, None, socket.AF_INET,
                    socket.SOCK_STREAM
                )
                if result:
                    ip = result[0][4][0]
                    self.matches.append(IOCMatch(
                        ioc=ioc,
                        location="DNS resolution",
                        context=f"{ioc.value} resolves to {ip}",
                    ))
            except (socket.gaierror, socket.timeout, OSError):
                pass  # Domain doesn't resolve - good

    def scan_file_contents(self, directory: str,
                           extensions: tuple = ('.py', '.js', '.sh', '.php',
                                                '.rb', '.pl', '.ps1')) -> None:
        """Scan file contents for IOC patterns (regex)."""
        print(f"[*] Scanning file contents in {directory}...")
        regex_iocs = self.ioc_db.get_by_type("REGEX")

        if not regex_iocs:
            return

        compiled_patterns = []
        for ioc in regex_iocs:
            try:
                compiled_patterns.append((re.compile(ioc.value, re.IGNORECASE), ioc))
            except re.error:
                print(f"    [!] Invalid regex pattern: {ioc.value}")

        scan_path = Path(directory)
        for filepath in scan_path.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in extensions:
                continue
            try:
                if filepath.stat().st_size > 10 * 1024 * 1024:  # Skip >10MB
                    continue
                content = filepath.read_text(errors='ignore')

                for pattern, ioc in compiled_patterns:
                    matches = pattern.findall(content)
                    if matches:
                        self.matches.append(IOCMatch(
                            ioc=ioc,
                            location=str(filepath),
                            context=f"Pattern '{ioc.value}' found {len(matches)} times",
                        ))

            except (PermissionError, OSError):
                pass

    def print_report(self) -> None:
        """Print IOC scan results."""
        print("\n" + "=" * 65)
        print("  IOC SCAN REPORT")
        print("=" * 65)
        print(f"  IOCs in database: {len(self.ioc_db.iocs)}")
        print(f"  Matches found: {len(self.matches)}")

        if not self.matches:
            print("\n  No IOC matches found. System appears clean.")
            print("=" * 65)
            return

        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            matches = [m for m in self.matches if m.ioc.severity == severity]
            if not matches:
                continue

            print(f"\n{'─' * 65}")
            print(f"  {severity} ({len(matches)} matches)")
            print(f"{'─' * 65}")

            for match in matches:
                print(f"\n  Type: {match.ioc.ioc_type}")
                print(f"  IOC:  {match.ioc.value}")
                print(f"  Desc: {match.ioc.description}")
                print(f"  Found: {match.location}")
                print(f"  Context: {match.context}")

        print(f"\n{'=' * 65}")
        print("  RECOMMENDED ACTIONS:")
        critical = [m for m in self.matches if m.ioc.severity == 'CRITICAL']
        if critical:
            print("  [!] CRITICAL matches found - initiate incident response")
            print("  [!] Isolate affected system immediately")
        elif self.matches:
            print("  [*] Review matches and determine if they are true positives")
            print("  [*] Escalate confirmed matches to security team")
        print("=" * 65)


# ─── Example IOC JSON format ───
SAMPLE_IOC_JSON = """
{
  "iocs": [
    {
      "type": "IP",
      "value": "198.51.100.1",
      "description": "Known C2 server for BotnetX",
      "source": "ThreatIntel Feed Alpha",
      "severity": "HIGH"
    },
    {
      "type": "DOMAIN",
      "value": "evil-malware.example.com",
      "description": "Malware distribution domain",
      "source": "OSINT",
      "severity": "CRITICAL"
    },
    {
      "type": "HASH_SHA256",
      "value": "a1b2c3d4e5f6...",
      "description": "Ransomware binary",
      "source": "VirusTotal",
      "severity": "CRITICAL"
    },
    {
      "type": "FILENAME",
      "value": "mimikatz.exe",
      "description": "Credential dumping tool",
      "source": "MITRE ATT&CK",
      "severity": "CRITICAL"
    },
    {
      "type": "REGEX",
      "value": "(?:eval|exec)\\\\s*\\\\(\\\\s*base64",
      "description": "Obfuscated code execution pattern",
      "source": "Custom rule",
      "severity": "HIGH"
    }
  ]
}
"""


if __name__ == "__main__":
    import sys

    # Initialize IOC database
    ioc_db = IOCDatabase()

    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        count = ioc_db.load_from_json(sys.argv[1])
        print(f"[*] Loaded {count} IOCs from {sys.argv[1]}")
        scan_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
    else:
        ioc_db.load_sample_iocs()
        print(f"[*] Loaded {len(ioc_db.iocs)} sample IOCs")
        scan_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

    # Run scanner
    scanner = IOCScanner(ioc_db)
    scanner.scan_all(scan_dir)
    scanner.print_report()
```

---

## 5. Digital Forensics Basics

### 5.1 Forensic Principles

```
┌──────────────────────────────────────────────────────────────────┐
│                  Digital Forensics Principles                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. PRESERVE the evidence                                        │
│     - Never work on original evidence                            │
│     - Create forensic images (bit-for-bit copies)               │
│     - Use write-blockers for disk access                         │
│     - Document everything with timestamps                        │
│                                                                   │
│  2. DOCUMENT the chain of custody                                │
│     - Who collected the evidence?                                │
│     - When was it collected?                                     │
│     - How was it stored?                                         │
│     - Who had access?                                            │
│                                                                   │
│  3. VERIFY integrity                                             │
│     - Hash all evidence immediately (SHA-256)                    │
│     - Verify hashes before and after analysis                    │
│     - Any change invalidates the evidence                        │
│                                                                   │
│  4. ANALYZE on copies                                            │
│     - Work on forensic copies, never originals                   │
│     - Use forensic tools that don't modify evidence              │
│     - Keep detailed notes of all analysis steps                  │
│                                                                   │
│  5. REPORT findings                                              │
│     - Factual, objective reporting                               │
│     - Reproducible methodology                                   │
│     - Clear chain from evidence to conclusions                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Order of Volatility

When collecting evidence, start with the most volatile (shortest-lived) data first.

```
┌──────────────────────────────────────────────────────────────────┐
│               Order of Volatility (Most → Least)                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Most Volatile (collect FIRST)                                   │
│  │                                                                │
│  ├── 1. CPU registers, cache                                     │
│  │      Lifetime: nanoseconds                                    │
│  │                                                                │
│  ├── 2. Memory (RAM)                                             │
│  │      Lifetime: power cycle                                    │
│  │      Contains: running processes, network connections,        │
│  │      decrypted data, passwords, encryption keys               │
│  │                                                                │
│  ├── 3. Network state                                            │
│  │      Lifetime: seconds-minutes                                │
│  │      Contains: active connections, routing tables, ARP cache  │
│  │                                                                │
│  ├── 4. Running processes                                        │
│  │      Lifetime: until process ends                             │
│  │      Contains: process list, open files, loaded libraries     │
│  │                                                                │
│  ├── 5. Disk (file system)                                       │
│  │      Lifetime: until overwritten                              │
│  │      Contains: files, logs, swap, temp files, slack space     │
│  │                                                                │
│  ├── 6. Remote logging / monitoring                              │
│  │      Lifetime: retention policy                               │
│  │                                                                │
│  └── 7. Archival media (backups, tapes)                          │
│         Lifetime: years                                           │
│                                                                   │
│  Least Volatile (collect LAST)                                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 Chain of Custody

```
┌──────────────────────────────────────────────────────────────────┐
│                    Chain of Custody Form                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Case Number: IR-2025-0042                                       │
│  Evidence ID: EVD-001                                            │
│  Description: Dell Latitude 7420 laptop, S/N: ABC123DEF          │
│  Location Found: Office 302, Building A                          │
│                                                                   │
│  Evidence Hash (at collection):                                  │
│  SHA-256: a1b2c3d4e5f6789...                                    │
│                                                                   │
│  ┌──────────┬───────────────────┬────────────┬────────────────┐  │
│  │   Date   │  Released By      │ Received By│    Purpose     │  │
│  ├──────────┼───────────────────┼────────────┼────────────────┤  │
│  │ 01/15/25 │ Officer Smith     │ Analyst Lee│ Initial        │  │
│  │ 10:30 AM │ (Badge #1234)     │ (IR Team)  │ collection     │  │
│  ├──────────┼───────────────────┼────────────┼────────────────┤  │
│  │ 01/15/25 │ Analyst Lee       │ Evidence   │ Secure         │  │
│  │ 11:45 AM │ (IR Team)         │ Locker     │ storage        │  │
│  ├──────────┼───────────────────┼────────────┼────────────────┤  │
│  │ 01/16/25 │ Evidence Locker   │ Forensic   │ Disk           │  │
│  │ 09:00 AM │                   │ Analyst Kim│ imaging        │  │
│  ├──────────┼───────────────────┼────────────┼────────────────┤  │
│  │ 01/16/25 │ Forensic Analyst  │ Evidence   │ Return after   │  │
│  │ 05:00 PM │ Kim               │ Locker     │ imaging        │  │
│  └──────────┴───────────────────┴────────────┴────────────────┘  │
│                                                                   │
│  Notes:                                                           │
│  - Laptop was powered off when collected                         │
│  - Battery was removed to prevent accidental boot                │
│  - Disk imaged using FTK Imager, hash verified                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.4 Forensic Disk Imaging

```bash
# Create a forensic image using dd
# WARNING: Be VERY careful with dd - wrong parameters can destroy data

# Step 1: Identify the target disk
lsblk
# or
fdisk -l

# Step 2: Create forensic image (bit-for-bit copy)
# /dev/sdb = source (evidence drive, via write-blocker)
# evidence.dd = destination image file
sudo dd if=/dev/sdb of=evidence.dd bs=4096 conv=noerror,sync status=progress

# Step 3: Calculate hash of original and image
sha256sum /dev/sdb > original_hash.txt
sha256sum evidence.dd > image_hash.txt

# Step 4: Verify hashes match
diff original_hash.txt image_hash.txt

# Better alternative: dc3dd (forensic-focused dd)
sudo dc3dd if=/dev/sdb of=evidence.dd hash=sha256 log=imaging.log

# Alternative: FTK Imager (cross-platform, GUI)
# Creates E01 (Expert Witness) format with built-in hashing
```

---

## 6. Memory Forensics Concepts

### 6.1 Why Memory Forensics?

```
┌──────────────────────────────────────────────────────────────────┐
│                What Lives Only in Memory?                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────┐                 │
│  │  Things you can ONLY find in RAM:           │                 │
│  │                                              │                 │
│  │  - Running processes (including hidden)     │                 │
│  │  - Network connections                       │                 │
│  │  - Decryption keys                           │                 │
│  │  - Passwords (plaintext in memory)          │                 │
│  │  - Injected code (fileless malware)         │                 │
│  │  - Clipboard contents                        │                 │
│  │  - Chat messages (before saved to disk)     │                 │
│  │  - Encryption keys (full disk encryption)   │                 │
│  │  - Command history                           │                 │
│  │  - Unpacked/decrypted malware               │                 │
│  └─────────────────────────────────────────────┘                 │
│                                                                   │
│  Modern malware often operates entirely in memory                │
│  ("fileless malware") to avoid disk-based detection.             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Memory Acquisition

```bash
# Linux memory acquisition

# Method 1: /proc/kcore (requires root)
sudo dd if=/proc/kcore of=memory.raw bs=1M

# Method 2: LiME (Linux Memory Extractor) - preferred
# Load LiME kernel module
sudo insmod lime-$(uname -r).ko "path=memory.lime format=lime"

# Method 3: AVML (Microsoft's Linux memory acquisition tool)
sudo ./avml memory.lime

# Windows memory acquisition
# - FTK Imager (free, GUI)
# - WinPmem (command line)
# - Belkasoft RAM Capturer (free)

# macOS memory acquisition
# - osxpmem
sudo ./osxpmem -o memory.aff4
```

### 6.3 Volatility Framework (Overview)

```bash
# Volatility 3 - Memory forensics framework
# Install: pip install volatility3

# Identify the OS profile
vol -f memory.raw windows.info

# List running processes
vol -f memory.raw windows.pslist
vol -f memory.raw windows.pstree  # Tree view

# Find hidden processes
vol -f memory.raw windows.psscan

# Network connections
vol -f memory.raw windows.netscan

# Command history
vol -f memory.raw windows.cmdline

# DLL list for a specific process
vol -f memory.raw windows.dlllist --pid 1234

# Dump a specific process
vol -f memory.raw windows.memmap --pid 1234 --dump

# Registry analysis
vol -f memory.raw windows.registry.hivelist

# Linux memory analysis
vol -f memory.raw linux.pslist
vol -f memory.raw linux.bash  # Bash history
vol -f memory.raw linux.netstat
```

---

## 7. Network Forensics

### 7.1 Packet Capture Analysis

```python
"""
pcap_analyzer.py - Basic network forensics with packet capture analysis.
Requires: pip install scapy

WARNING: Only analyze captures from networks you are authorized to monitor.
"""

try:
    from scapy.all import rdpcap, IP, TCP, UDP, DNS, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Scapy not installed. Install with: pip install scapy")

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class PcapAnalysis:
    """Results of PCAP file analysis."""
    total_packets: int = 0
    protocols: dict = field(default_factory=dict)
    top_talkers: list = field(default_factory=list)
    dns_queries: list = field(default_factory=list)
    suspicious_connections: list = field(default_factory=list)
    http_requests: list = field(default_factory=list)
    large_transfers: list = field(default_factory=list)


def analyze_pcap(filepath: str) -> PcapAnalysis:
    """
    Analyze a PCAP file for security-relevant information.

    Args:
        filepath: Path to .pcap or .pcapng file

    Returns:
        PcapAnalysis with findings
    """
    if not SCAPY_AVAILABLE:
        raise ImportError("Scapy is required for PCAP analysis")

    packets = rdpcap(filepath)
    analysis = PcapAnalysis(total_packets=len(packets))

    # Track statistics
    ip_src_counter = Counter()
    ip_dst_counter = Counter()
    protocol_counter = Counter()
    connection_sizes = defaultdict(int)
    dns_queries = []
    http_requests = []

    for pkt in packets:
        # Protocol analysis
        if pkt.haslayer(TCP):
            protocol_counter['TCP'] += 1
        elif pkt.haslayer(UDP):
            protocol_counter['UDP'] += 1

        # IP layer analysis
        if pkt.haslayer(IP):
            src = pkt[IP].src
            dst = pkt[IP].dst
            ip_src_counter[src] += 1
            ip_dst_counter[dst] += 1

            # Track connection data volume
            if pkt.haslayer(Raw):
                key = f"{src} -> {dst}"
                connection_sizes[key] += len(pkt[Raw].load)

        # DNS analysis
        if pkt.haslayer(DNS) and pkt[DNS].qr == 0:  # Query
            try:
                query_name = pkt[DNS].qd.qname.decode('utf-8', errors='ignore')
                dns_queries.append({
                    'query': query_name,
                    'src': pkt[IP].src if pkt.haslayer(IP) else 'unknown',
                    'type': pkt[DNS].qd.qtype,
                })
            except (AttributeError, IndexError):
                pass

        # HTTP request detection (basic)
        if pkt.haslayer(TCP) and pkt.haslayer(Raw):
            payload = pkt[Raw].load
            try:
                text = payload.decode('utf-8', errors='ignore')
                if text.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ')):
                    lines = text.split('\r\n')
                    http_requests.append({
                        'method': lines[0].split(' ')[0],
                        'path': lines[0].split(' ')[1] if len(lines[0].split(' ')) > 1 else '',
                        'src': pkt[IP].src if pkt.haslayer(IP) else 'unknown',
                        'dst': pkt[IP].dst if pkt.haslayer(IP) else 'unknown',
                    })
            except (UnicodeDecodeError, IndexError):
                pass

    # Compile results
    analysis.protocols = dict(protocol_counter)

    # Top talkers (by packet count)
    analysis.top_talkers = [
        {'ip': ip, 'packets_sent': count}
        for ip, count in ip_src_counter.most_common(10)
    ]

    # DNS queries (deduplicated)
    seen_queries = set()
    for q in dns_queries:
        if q['query'] not in seen_queries:
            analysis.dns_queries.append(q)
            seen_queries.add(q['query'])

    # Large data transfers (potential exfiltration)
    analysis.large_transfers = [
        {'connection': conn, 'bytes': size}
        for conn, size in sorted(
            connection_sizes.items(), key=lambda x: x[1], reverse=True
        )[:10]
    ]

    analysis.http_requests = http_requests[:50]

    # Suspicious pattern detection
    analysis.suspicious_connections = detect_suspicious_patterns(
        packets, ip_src_counter, dns_queries
    )

    return analysis


def detect_suspicious_patterns(packets, ip_counter, dns_queries):
    """Detect suspicious network patterns."""
    suspicious = []

    # 1. Beaconing detection (regular interval connections)
    # Simplified: check for IPs with very regular packet intervals
    # (Real beaconing detection requires statistical analysis)

    # 2. DNS tunneling indicators
    long_queries = [q for q in dns_queries if len(q['query']) > 50]
    if long_queries:
        suspicious.append({
            'type': 'DNS_TUNNELING_POSSIBLE',
            'description': f"Found {len(long_queries)} unusually long DNS queries",
            'samples': [q['query'][:80] for q in long_queries[:3]],
        })

    # 3. Port scanning detection
    dst_ports = defaultdict(set)
    for pkt in packets:
        if pkt.haslayer(TCP) and pkt.haslayer(IP):
            src = pkt[IP].src
            dst_port = pkt[TCP].dport
            dst_ports[src].add(dst_port)

    for ip, ports in dst_ports.items():
        if len(ports) > 20:  # Hitting many different ports
            suspicious.append({
                'type': 'PORT_SCAN_POSSIBLE',
                'description': f"{ip} contacted {len(ports)} different ports",
                'samples': sorted(list(ports))[:10],
            })

    return suspicious


def print_pcap_report(analysis: PcapAnalysis) -> None:
    """Print formatted PCAP analysis report."""
    print("=" * 65)
    print("  NETWORK FORENSICS REPORT")
    print("=" * 65)
    print(f"  Total packets: {analysis.total_packets}")
    print(f"  Protocols: {analysis.protocols}")

    print(f"\n{'─' * 65}")
    print("  TOP TALKERS (by packets sent)")
    print(f"{'─' * 65}")
    for t in analysis.top_talkers:
        print(f"  {t['ip']:20s} {t['packets_sent']} packets")

    if analysis.dns_queries:
        print(f"\n{'─' * 65}")
        print(f"  DNS QUERIES ({len(analysis.dns_queries)} unique)")
        print(f"{'─' * 65}")
        for q in analysis.dns_queries[:20]:
            print(f"  {q['src']:20s} -> {q['query']}")

    if analysis.large_transfers:
        print(f"\n{'─' * 65}")
        print("  LARGEST DATA TRANSFERS")
        print(f"{'─' * 65}")
        for t in analysis.large_transfers:
            size_kb = t['bytes'] / 1024
            print(f"  {t['connection']:40s} {size_kb:.1f} KB")

    if analysis.suspicious_connections:
        print(f"\n{'─' * 65}")
        print("  SUSPICIOUS PATTERNS")
        print(f"{'─' * 65}")
        for s in analysis.suspicious_connections:
            print(f"  [{s['type']}] {s['description']}")
            if s.get('samples'):
                for sample in s['samples']:
                    print(f"    - {sample}")

    print(f"\n{'=' * 65}")
```

### 7.2 Useful Command-Line Tools

```bash
# tcpdump - capture packets
# Capture all traffic on eth0
sudo tcpdump -i eth0 -w capture.pcap

# Capture only traffic to/from specific IP
sudo tcpdump -i eth0 host 192.168.1.100 -w suspicious.pcap

# Capture only HTTP traffic
sudo tcpdump -i eth0 port 80 -w http_traffic.pcap

# Capture DNS traffic
sudo tcpdump -i eth0 port 53 -w dns_traffic.pcap

# tshark (command-line Wireshark)
# Extract HTTP requests
tshark -r capture.pcap -Y "http.request" -T fields \
  -e ip.src -e http.request.method -e http.request.uri

# Extract DNS queries
tshark -r capture.pcap -Y "dns.qr == 0" -T fields \
  -e ip.src -e dns.qry.name

# Extract file transfers
tshark -r capture.pcap --export-objects http,exported_files/

# Show conversation statistics
tshark -r capture.pcap -z conv,ip -q
```

---

## 8. Incident Response Playbooks

### 8.1 Playbook Template

```
┌──────────────────────────────────────────────────────────────────┐
│                  IR Playbook Template                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PLAYBOOK: [Incident Type]                                       │
│  VERSION: 1.0                                                     │
│  LAST UPDATED: [Date]                                            │
│  OWNER: [Team/Person]                                            │
│  SEVERITY: [P1-P4]                                               │
│                                                                   │
│  TRIGGER:                                                         │
│  [What alerts/conditions activate this playbook]                 │
│                                                                   │
│  INITIAL TRIAGE (first 15 minutes):                              │
│  [ ] Step 1: ...                                                 │
│  [ ] Step 2: ...                                                 │
│  [ ] Step 3: Classify severity                                   │
│  [ ] Step 4: Notify incident manager                             │
│                                                                   │
│  CONTAINMENT (first 1-4 hours):                                  │
│  [ ] Step 1: ...                                                 │
│  [ ] Step 2: ...                                                 │
│  [ ] Step 3: Preserve evidence                                   │
│                                                                   │
│  ERADICATION:                                                     │
│  [ ] Step 1: ...                                                 │
│  [ ] Step 2: ...                                                 │
│                                                                   │
│  RECOVERY:                                                        │
│  [ ] Step 1: ...                                                 │
│  [ ] Step 2: Verify normal operations                            │
│                                                                   │
│  COMMUNICATION:                                                   │
│  - Internal: [who to notify and when]                            │
│  - External: [customers, regulators, law enforcement]            │
│                                                                   │
│  ESCALATION:                                                      │
│  - Condition → Action                                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 Playbook: Ransomware Incident

```
┌──────────────────────────────────────────────────────────────────┐
│            PLAYBOOK: Ransomware Incident                          │
├──────────────────────────────────────────────────────────────────┤
│  SEVERITY: P1 (Critical)                                         │
│                                                                   │
│  TRIGGER:                                                         │
│  - Ransom note displayed on endpoint                             │
│  - Mass file encryption detected                                  │
│  - EDR alert: ransomware behavior                                │
│                                                                   │
│  INITIAL TRIAGE (first 15 minutes):                              │
│  [ ] 1. DO NOT power off affected systems                        │
│  [ ] 2. Disconnect affected systems from network                 │
│         (pull ethernet, disable WiFi - do NOT shut down)        │
│  [ ] 3. Document ransom note (photograph/screenshot)             │
│  [ ] 4. Identify ransomware variant if possible                  │
│  [ ] 5. Determine scope: how many systems affected?              │
│  [ ] 6. Notify incident manager → activate IR team               │
│  [ ] 7. Notify CISO / executive management                      │
│                                                                   │
│  CONTAINMENT (first 1-4 hours):                                  │
│  [ ] 1. Isolate affected network segments                        │
│  [ ] 2. Block known ransomware C2 IPs/domains                   │
│  [ ] 3. Disable network shares to prevent spread                 │
│  [ ] 4. Reset all potentially compromised credentials            │
│  [ ] 5. Capture memory dumps of affected systems                 │
│  [ ] 6. Preserve logs (SIEM, firewall, endpoint)                │
│  [ ] 7. Check backup integrity (are backups affected?)           │
│                                                                   │
│  ERADICATION:                                                     │
│  [ ] 1. Identify initial infection vector (email, exploit, etc.) │
│  [ ] 2. Check NoMoreRansom.org for decryption tools              │
│  [ ] 3. Remove malware from all affected systems                 │
│  [ ] 4. Patch vulnerability that allowed infection               │
│  [ ] 5. Scan all systems for persistence mechanisms              │
│                                                                   │
│  RECOVERY:                                                        │
│  [ ] 1. Restore from clean, verified backups                     │
│  [ ] 2. Rebuild systems that cannot be cleaned                   │
│  [ ] 3. Restore in phases, monitoring for reinfection            │
│  [ ] 4. Reset all passwords organization-wide                    │
│  [ ] 5. Enhance monitoring for 30 days post-recovery            │
│                                                                   │
│  COMMUNICATION:                                                   │
│  - Internal: All-hands notification within 2 hours               │
│  - Legal: Engage legal counsel immediately                        │
│  - Insurance: Notify cyber insurance carrier                     │
│  - Law enforcement: File report with FBI IC3                     │
│  - Regulators: Per regulatory requirements (GDPR: 72 hours)     │
│  - Customers: If data breach confirmed                           │
│                                                                   │
│  DO NOT:                                                          │
│  - Pay the ransom without consulting legal and law enforcement   │
│  - Communicate with attackers without legal guidance              │
│  - Destroy evidence                                              │
│  - Restore from backups before ensuring they are clean           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 8.3 Playbook: Compromised Credentials

```
┌──────────────────────────────────────────────────────────────────┐
│         PLAYBOOK: Compromised Credentials                         │
├──────────────────────────────────────────────────────────────────┤
│  SEVERITY: P2 (High)                                             │
│                                                                   │
│  TRIGGER:                                                         │
│  - Credential found on dark web / paste site                     │
│  - User reports phishing / credential theft                      │
│  - Impossible travel alert                                        │
│  - MFA bypass detected                                           │
│                                                                   │
│  INITIAL TRIAGE:                                                  │
│  [ ] 1. Identify affected account(s)                             │
│  [ ] 2. Determine credential type (password, API key, token)    │
│  [ ] 3. Check for unauthorized access in audit logs              │
│  [ ] 4. Determine if MFA was enabled                             │
│                                                                   │
│  CONTAINMENT:                                                     │
│  [ ] 1. Force password reset on affected account                 │
│  [ ] 2. Revoke all active sessions / tokens                     │
│  [ ] 3. Rotate API keys if applicable                            │
│  [ ] 4. Enable MFA if not already enabled                        │
│  [ ] 5. Block suspicious source IPs                              │
│  [ ] 6. Check for mailbox rules (forwarding, deletion)          │
│                                                                   │
│  INVESTIGATION:                                                   │
│  [ ] 1. Review all actions taken with compromised credential    │
│  [ ] 2. Check for lateral movement (access to other systems)    │
│  [ ] 3. Check for data access / exfiltration                    │
│  [ ] 4. Identify how credential was compromised                  │
│  [ ] 5. Check if credential was reused on other services        │
│                                                                   │
│  RECOVERY:                                                        │
│  [ ] 1. Verify account is secured (new password + MFA)           │
│  [ ] 2. Reverse any unauthorized changes                         │
│  [ ] 3. Notify user of incident and require security training    │
│  [ ] 4. Monitor account for 30 days                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Post-Incident Review Template

```python
"""
incident_report.py - Generate post-incident review reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TimelineEvent:
    """A single event in the incident timeline."""
    timestamp: str
    description: str
    actor: str = ""        # Who performed the action
    evidence: str = ""     # Supporting evidence


@dataclass
class ActionItem:
    """A follow-up action from the incident review."""
    description: str
    owner: str
    due_date: str
    priority: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    status: str = "OPEN"      # OPEN, IN_PROGRESS, DONE


@dataclass
class IncidentReport:
    """Complete post-incident review report."""
    # Metadata
    incident_id: str
    title: str
    severity: str
    status: str = "CLOSED"
    report_date: str = ""
    report_author: str = ""

    # Timeline
    detected_at: str = ""
    contained_at: str = ""
    eradicated_at: str = ""
    recovered_at: str = ""
    closed_at: str = ""

    # Details
    summary: str = ""
    root_cause: str = ""
    impact: str = ""
    affected_systems: list[str] = field(default_factory=list)
    affected_users: int = 0
    data_compromised: str = ""

    # Analysis
    attack_vector: str = ""
    attacker_info: str = ""
    timeline: list[TimelineEvent] = field(default_factory=list)

    # Lessons
    what_went_well: list[str] = field(default_factory=list)
    what_went_wrong: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)

    # Metrics
    time_to_detect: str = ""       # Time from compromise to detection
    time_to_contain: str = ""      # Time from detection to containment
    time_to_recover: str = ""      # Time from containment to recovery
    total_duration: str = ""       # Total incident duration

    def __post_init__(self):
        if not self.report_date:
            self.report_date = datetime.now().strftime("%Y-%m-%d")

    def generate_markdown(self) -> str:
        """Generate a Markdown report."""
        lines = []

        lines.append(f"# Incident Report: {self.incident_id}")
        lines.append(f"\n**Title**: {self.title}")
        lines.append(f"**Severity**: {self.severity}")
        lines.append(f"**Status**: {self.status}")
        lines.append(f"**Report Date**: {self.report_date}")
        lines.append(f"**Author**: {self.report_author}")

        # Executive Summary
        lines.append("\n## Executive Summary\n")
        lines.append(self.summary)

        # Impact
        lines.append("\n## Impact\n")
        lines.append(self.impact)
        if self.affected_systems:
            lines.append(f"\n**Affected Systems**: {', '.join(self.affected_systems)}")
        lines.append(f"**Affected Users**: {self.affected_users}")
        if self.data_compromised:
            lines.append(f"**Data Compromised**: {self.data_compromised}")

        # Timeline
        lines.append("\n## Timeline\n")
        lines.append("| Time | Event | Actor |")
        lines.append("|------|-------|-------|")
        for event in self.timeline:
            lines.append(
                f"| {event.timestamp} | {event.description} | {event.actor} |"
            )

        # Key Metrics
        lines.append("\n## Key Metrics\n")
        lines.append(f"- **Time to Detect**: {self.time_to_detect}")
        lines.append(f"- **Time to Contain**: {self.time_to_contain}")
        lines.append(f"- **Time to Recover**: {self.time_to_recover}")
        lines.append(f"- **Total Duration**: {self.total_duration}")

        # Root Cause
        lines.append("\n## Root Cause Analysis\n")
        lines.append(self.root_cause)
        lines.append(f"\n**Attack Vector**: {self.attack_vector}")

        # Lessons Learned
        lines.append("\n## Lessons Learned\n")
        lines.append("### What Went Well\n")
        for item in self.what_went_well:
            lines.append(f"- {item}")
        lines.append("\n### What Needs Improvement\n")
        for item in self.what_went_wrong:
            lines.append(f"- {item}")

        # Action Items
        lines.append("\n## Action Items\n")
        lines.append("| # | Action | Owner | Due Date | Priority | Status |")
        lines.append("|---|--------|-------|----------|----------|--------|")
        for i, action in enumerate(self.action_items, 1):
            lines.append(
                f"| {i} | {action.description} | {action.owner} | "
                f"{action.due_date} | {action.priority} | {action.status} |"
            )

        return "\n".join(lines)


# ─── Example Usage ───

def create_sample_report() -> IncidentReport:
    """Create a sample incident report for demonstration."""
    report = IncidentReport(
        incident_id="IR-2025-0042",
        title="Unauthorized Access via Compromised API Key",
        severity="P2 - High",
        report_author="Security Team",
        summary=(
            "On January 15, 2025, an unauthorized party accessed our "
            "production API using a compromised API key. The key was "
            "inadvertently committed to a public GitHub repository. "
            "The attacker accessed customer order data for approximately "
            "2 hours before detection."
        ),
        root_cause=(
            "A developer committed an API key to a public GitHub "
            "repository on January 10. The key was scraped by an "
            "automated bot and used to access the production API "
            "on January 15. Pre-commit hooks for secret detection "
            "were not configured on the developer's machine."
        ),
        impact=(
            "Customer order data (names, addresses, order history) "
            "for approximately 1,200 customers was potentially accessed. "
            "No payment card data was exposed (stored separately). "
            "No evidence of data modification."
        ),
        attack_vector="Compromised API key from public Git repository",
        affected_systems=["api-prod-01", "api-prod-02", "orders-db"],
        affected_users=1200,
        data_compromised="Customer names, addresses, order history",
        detected_at="2025-01-15 14:30 UTC",
        contained_at="2025-01-15 14:45 UTC",
        eradicated_at="2025-01-15 16:00 UTC",
        recovered_at="2025-01-15 18:00 UTC",
        closed_at="2025-01-20 09:00 UTC",
        time_to_detect="5 days (from key commit to detection)",
        time_to_contain="15 minutes",
        time_to_recover="3.5 hours",
        total_duration="5 days",
        timeline=[
            TimelineEvent(
                "2025-01-10 09:15", "API key committed to public repo",
                "Developer A"
            ),
            TimelineEvent(
                "2025-01-15 12:30", "First unauthorized API access",
                "Unknown attacker"
            ),
            TimelineEvent(
                "2025-01-15 14:30", "Anomalous API usage alert triggered",
                "SIEM"
            ),
            TimelineEvent(
                "2025-01-15 14:35", "SOC analyst confirms unauthorized access",
                "Analyst B"
            ),
            TimelineEvent(
                "2025-01-15 14:45", "API key revoked, attacker blocked",
                "Analyst B"
            ),
            TimelineEvent(
                "2025-01-15 15:00", "Incident manager notified, IR activated",
                "IR Lead C"
            ),
            TimelineEvent(
                "2025-01-15 16:00", "All exposed API keys rotated",
                "DevOps Team"
            ),
            TimelineEvent(
                "2025-01-15 18:00", "Monitoring confirms no further access",
                "SOC Team"
            ),
        ],
        what_went_well=[
            "SIEM alert fired quickly once anomalous pattern detected",
            "API key revocation was fast (15 min from alert to containment)",
            "IR team followed playbook effectively",
            "Good communication between SOC and development teams",
        ],
        what_went_wrong=[
            "API key was in public repo for 5 days before detection",
            "No automated secret scanning on GitHub repositories",
            "Pre-commit hooks not enforced across all developer machines",
            "API key had overly broad permissions (read all orders)",
            "No IP-based access restrictions on API keys",
        ],
        action_items=[
            ActionItem(
                "Deploy Gitleaks on all repositories",
                "DevOps Team", "2025-02-01", "HIGH"
            ),
            ActionItem(
                "Enforce pre-commit hooks with detect-secrets",
                "Dev Lead", "2025-02-15", "HIGH"
            ),
            ActionItem(
                "Implement API key scope restrictions (least privilege)",
                "API Team", "2025-03-01", "HIGH"
            ),
            ActionItem(
                "Add IP allowlisting for production API keys",
                "Infrastructure", "2025-03-01", "MEDIUM"
            ),
            ActionItem(
                "Conduct developer security training (secrets management)",
                "Security Team", "2025-02-28", "MEDIUM"
            ),
            ActionItem(
                "Implement automated key rotation (90-day max)",
                "DevOps Team", "2025-04-01", "MEDIUM"
            ),
        ],
    )

    return report


if __name__ == "__main__":
    report = create_sample_report()
    markdown = report.generate_markdown()
    print(markdown)

    # Save to file
    with open("incident_report_IR-2025-0042.md", "w") as f:
        f.write(markdown)
    print("\nReport saved to: incident_report_IR-2025-0042.md")
```

---

## 10. Exercises

### Exercise 1: Log Analysis

Given the following sample log entries, identify all security incidents:

```
192.168.1.50 - - [15/Jan/2025:10:00:01 +0000] "GET /login HTTP/1.1" 200 1234
192.168.1.50 - - [15/Jan/2025:10:00:02 +0000] "POST /login HTTP/1.1" 401 89
192.168.1.50 - - [15/Jan/2025:10:00:03 +0000] "POST /login HTTP/1.1" 401 89
192.168.1.50 - - [15/Jan/2025:10:00:04 +0000] "POST /login HTTP/1.1" 401 89
10.0.0.5 - admin [15/Jan/2025:10:05:00 +0000] "GET /admin/users HTTP/1.1" 200 5678
10.0.0.5 - admin [15/Jan/2025:10:05:01 +0000] "GET /admin/export?table=users HTTP/1.1" 200 890123
10.0.0.5 - admin [15/Jan/2025:10:05:02 +0000] "GET /admin/export?table=payments HTTP/1.1" 200 1234567
203.0.113.10 - - [15/Jan/2025:10:10:00 +0000] "GET /../../etc/passwd HTTP/1.1" 403 0
203.0.113.10 - - [15/Jan/2025:10:10:01 +0000] "GET /search?q=' OR 1=1 -- HTTP/1.1" 500 0
203.0.113.10 - - [15/Jan/2025:10:10:02 +0000] "GET /search?q=<script>alert(1)</script> HTTP/1.1" 200 456
```

**Tasks:**
1. Classify each suspicious pattern (brute force, traversal, SQLi, XSS, data exfiltration)
2. Determine the severity of each finding
3. Write a brief incident summary for each

### Exercise 2: IOC Database

Create a JSON file with at least 20 IOCs covering:
- 5 malicious IP addresses
- 5 malicious domains
- 5 malware file hashes
- 5 suspicious filenames

Run the IOC scanner against a test directory you create with some matching items.

### Exercise 3: Incident Response Playbook

Write a complete incident response playbook for a **SQL injection attack** that:
1. Defines trigger conditions (what alerts indicate SQLi)
2. Covers all four NIST phases
3. Includes specific commands/tools to use at each step
4. Defines communication and escalation procedures
5. Includes a post-incident checklist

### Exercise 4: Memory Forensics Analysis

Research Volatility 3 and write a step-by-step guide for analyzing a memory dump to:
1. List all running processes and identify suspicious ones
2. Find active network connections
3. Extract command line arguments for each process
4. Identify injected DLLs or code
5. Recover encryption keys or passwords from memory

### Exercise 5: Post-Incident Report

Using the `IncidentReport` class from this lesson, create a complete post-incident report for the following scenario:

> Your company's web application was defaced at 3 AM on a Sunday. The attacker exploited a known CVE in an unpatched WordPress plugin. They replaced the homepage with a political message. Your monitoring system detected the change at 3:15 AM. The on-call engineer restored from backup at 4:00 AM. Investigation revealed the attacker also created a backdoor admin account.

### Exercise 6: PCAP Analysis

Download a sample PCAP file from a CTF or security training resource (e.g., malware-traffic-analysis.net). Analyze it using the tools from this lesson and write a network forensics report covering:
1. Top talkers (most active IP addresses)
2. DNS queries (especially suspicious ones)
3. HTTP requests (look for malware downloads, C2 communications)
4. Any indicators of data exfiltration

---

## Summary

```
┌──────────────────────────────────────────────────────────────────┐
│           Incident Response Key Takeaways                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Preparation is everything: Have plans, tools, and trained   │
│     people BEFORE an incident occurs                             │
│                                                                   │
│  2. Follow the NIST lifecycle: Preparation → Detection →        │
│     Containment → Eradication → Recovery → Lessons Learned      │
│                                                                   │
│  3. Preserve evidence: Document everything, maintain chain      │
│     of custody, hash all evidence, work on copies                │
│                                                                   │
│  4. Centralize logging: You cannot investigate what you          │
│     did not log. Invest in logging infrastructure                │
│                                                                   │
│  5. Automate detection: Use SIEM correlation rules and          │
│     IOC scanning to reduce detection time                        │
│                                                                   │
│  6. Practice with playbooks: Written, tested playbooks reduce   │
│     response time and ensure consistency                         │
│                                                                   │
│  7. Learn from incidents: Post-incident reviews are the most    │
│     valuable source of security improvements                     │
│                                                                   │
│  8. Time matters: Minutes count during active incidents.         │
│     Mean Time to Detect (MTTD) and Mean Time to Respond         │
│     (MTTR) are your key metrics                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [13. Security Testing](13_Security_Testing.md) | **Next**: [15. Project: Building a Secure REST API](15_Project_Secure_API.md)
