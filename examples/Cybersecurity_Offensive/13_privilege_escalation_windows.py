"""
Example: Privilege Escalation — Windows
=========================================
Unquoted service path detector, token privilege analyzer, registry ACL
checker, and AlwaysInstallElevated check.

IMPORTANT: For authorized security testing and CTF only.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Unquoted Service Path Detection
# ---------------------------------------------------------------------------

def find_unquoted_paths(services: list[dict]) -> list[dict]:
    """Identify services with unquoted paths containing spaces."""
    results = []
    for svc in services:
        path = svc["path"]
        # Skip if already quoted
        if path.startswith('"'):
            results.append({"service": svc["name"], "vulnerable": False,
                            "hijack_paths": []})
            continue
        # Check for spaces in path (before the .exe)
        exe_idx = path.lower().find(".exe")
        if exe_idx < 0:
            continue
        path_to_exe = path[:exe_idx + 4]
        if " " not in path_to_exe:
            results.append({"service": svc["name"], "vulnerable": False,
                            "hijack_paths": []})
            continue

        # Generate hijack paths at each space boundary
        hijack_paths = []
        parts = path_to_exe.split("\\")
        accumulated = ""
        for i, part in enumerate(parts):
            if i == 0:
                accumulated = part
                continue
            if " " in part:
                space_idx = part.index(" ")
                hijack = accumulated + "\\" + part[:space_idx] + ".exe"
                hijack_paths.append(hijack)
            accumulated += "\\" + part

        results.append({"service": svc["name"], "vulnerable": True,
                        "hijack_paths": hijack_paths,
                        "run_as": svc.get("run_as", "Unknown")})
    return results


# ---------------------------------------------------------------------------
# Token Privilege Analysis
# ---------------------------------------------------------------------------

DANGEROUS_PRIVILEGES = {
    "SeImpersonatePrivilege": {
        "risk": "critical",
        "technique": "Potato attacks (JuicyPotato, PrintSpoofer, GodPotato)",
        "description": "Impersonate any token, often leads to SYSTEM",
    },
    "SeAssignPrimaryTokenPrivilege": {
        "risk": "critical",
        "technique": "Token manipulation to create SYSTEM process",
        "description": "Assign primary token to new process",
    },
    "SeDebugPrivilege": {
        "risk": "critical",
        "technique": "Inject into SYSTEM processes (e.g., lsass.exe)",
        "description": "Debug any process regardless of ACL",
    },
    "SeBackupPrivilege": {
        "risk": "high",
        "technique": "Read any file (SAM, SYSTEM hives)",
        "description": "Bypass ACLs for file read operations",
    },
    "SeRestorePrivilege": {
        "risk": "high",
        "technique": "Write to any file (DLL hijacking, service binary)",
        "description": "Bypass ACLs for file write operations",
    },
    "SeTakeOwnershipPrivilege": {
        "risk": "high",
        "technique": "Take ownership of system files, then modify",
        "description": "Take ownership of any securable object",
    },
    "SeLoadDriverPrivilege": {
        "risk": "high",
        "technique": "Load malicious kernel driver",
        "description": "Load/unload device drivers",
    },
}


def analyze_token_privileges(privileges: list[dict]) -> list[dict]:
    """Analyze process token privileges for escalation potential."""
    findings = []
    for priv in privileges:
        name = priv["name"]
        if name in DANGEROUS_PRIVILEGES:
            info = DANGEROUS_PRIVILEGES[name]
            findings.append({
                "privilege": name,
                "enabled": priv["enabled"],
                "risk": info["risk"],
                "technique": info["technique"],
                "actionable": priv["enabled"],
            })
    return findings


# ---------------------------------------------------------------------------
# Windows Service Binary Permissions
# ---------------------------------------------------------------------------

@dataclass
class ServicePermCheck:
    service_name: str
    binary_path: str
    service_acl_writable: bool
    binary_writable: bool
    dll_hijack_possible: bool
    findings: list[str] = field(default_factory=list)

    def analyze(self) -> "ServicePermCheck":
        if self.service_acl_writable:
            self.findings.append(
                "Service configuration is writable — change binary path")
        if self.binary_writable:
            self.findings.append(
                "Service binary is writable — replace with payload")
        if self.dll_hijack_possible:
            self.findings.append(
                "Missing DLL in search path — plant malicious DLL")
        return self


# ---------------------------------------------------------------------------
# Common Windows Privesc Checks
# ---------------------------------------------------------------------------

WINDOWS_CHECKS = [
    {"check": "Unquoted Service Paths",
     "command": 'wmic service get name,pathname | findstr /i /v "C:\\Windows"',
     "risk": "medium"},
    {"check": "AlwaysInstallElevated",
     "command": "reg query HKCU\\SOFTWARE\\Policies\\Microsoft\\Windows\\Installer /v AlwaysInstallElevated",
     "risk": "high"},
    {"check": "Stored Credentials",
     "command": "cmdkey /list",
     "risk": "high"},
    {"check": "Scheduled Tasks (writable)",
     "command": "schtasks /query /fo CSV /v | findstr /i writable",
     "risk": "medium"},
    {"check": "Token Privileges",
     "command": "whoami /priv",
     "risk": "varies"},
    {"check": "Autologon Credentials",
     "command": "reg query HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon",
     "risk": "high"},
    {"check": "Unattended Install Files",
     "command": "dir /s C:\\unattend.xml C:\\sysprep.xml 2>nul",
     "risk": "high"},
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Privilege Escalation — Windows Examples")
    print("=" * 50)

    # Unquoted service paths
    services = [
        {"name": "VulnSvc", "run_as": "LocalSystem",
         "path": "C:\\Program Files\\Vulnerable App\\service.exe"},
        {"name": "SafeSvc", "run_as": "LocalSystem",
         "path": '"C:\\Program Files\\Safe App\\service.exe"'},
        {"name": "NoSpace", "run_as": "LocalSystem",
         "path": "C:\\tools\\myservice.exe"},
    ]
    print("\nUnquoted Service Path Check:")
    for result in find_unquoted_paths(services):
        status = "VULNERABLE" if result["vulnerable"] else "safe"
        print(f"  [{status:10s}] {result['service']}")
        for hp in result.get("hijack_paths", []):
            print(f"    Try: {hp}")

    # Token privileges
    print("\nDangerous Token Privileges:")
    privs = [
        {"name": "SeImpersonatePrivilege", "enabled": True},
        {"name": "SeDebugPrivilege", "enabled": False},
        {"name": "SeBackupPrivilege", "enabled": True},
        {"name": "SeChangeNotifyPrivilege", "enabled": True},
    ]
    for finding in analyze_token_privileges(privs):
        enabled = "ENABLED" if finding["actionable"] else "disabled"
        print(f"  [{finding['risk']:8s}] {finding['privilege']} ({enabled})")
        print(f"    -> {finding['technique']}")

    # Enumeration checklist
    print("\nWindows Privesc Enumeration Checklist:")
    for check in WINDOWS_CHECKS:
        print(f"  [{check['risk']:7s}] {check['check']}")
        print(f"    CMD: {check['command'][:60]}...")


if __name__ == "__main__":
    demo()
